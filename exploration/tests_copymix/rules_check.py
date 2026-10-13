"""Quick look (not a pre-registered test): outgoing-boat's two fixed rules on our held-out F.
R1 nm-line: rows without an NM record (FLIGHT_ID missing) with MVT-SCHED > 6 h -> per-airport y = a + b*ms (fit on other months).
R2 Rome: LIRF, no NM record, 15 h < MVT-SCHED <= 30 h -> y = a + b*ms (fit on other months)."""
import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'tests_copymix')
import numpy as np, pandas as pd
from common import dataset as D
from common import io as cio
from copymix import times, lobt_clip
raw = cio.load_training(columns=["MVT_ID_mvt","PHASE_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","LOBT_flt","FLIGHT_ID_mvt"])
raw = raw[raw.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
X,y,_ = D.load_part("full","train","with_aobt",filter_target=False); y=pd.Series(np.asarray(y,float),index=X.index)
d = pd.read_parquet('tests_copymix/cache/F_heldout.parquet'); d["y"]=y.reindex(d.index).to_numpy(); d=d[np.isfinite(d.y)]
mv,sc,lb = times(d.index, raw); d["ms"]=mv-sc; d["mv"]=mv; d["lb"]=lb
d["nm"]=raw.FLIGHT_ID_mvt.reindex(d.index).isna().to_numpy()
d["Fc"]=lobt_clip(d.F.to_numpy(),mv,lb)
rm=lambda r: float(np.sqrt(np.mean(np.square(r))))
def line(ms,yy):
    A=np.c_[np.ones(len(ms)),ms]; return np.linalg.lstsq(A,yy,rcond=None)[0]
for name,sel_fn in (("R1 nm-line >6h", lambda t: t.nm & (t.ms>6*3600)),
                    ("R2 Rome 15-30h", lambda t: t.nm & t.ap.eq("LIRF") & (t.ms>15*3600) & (t.ms<=30*3600))):
    print("==", name)
    sA=[];sB=[]
    for m in ["01","07","08","09","10","11","12"]:
        tr,te=d[d.mon!=m],d[d.mon==m]
        P=te.Fc.to_numpy().copy(); s=sel_fn(te).to_numpy(); st=sel_fn(tr).to_numpy()
        for a in te.ap[s].unique():
            k=st&tr.ap.eq(a).to_numpy()
            if k.sum()<20: continue
            ab=line(tr.ms.to_numpy()[k],tr.y.to_numpy()[k]); kk=s&te.ap.eq(a).to_numpy()
            P[kk]=ab[0]+ab[1]*te.ms.to_numpy()[kk]
        P=lobt_clip(P,te.mv.to_numpy(),te.lb.to_numpy())
        a_=(te.y-te.Fc).to_numpy()**2; b_=(te.y.to_numpy()-P)**2; sA.append(a_); sB.append(b_)
        print(f"  {m}: rows {int(s.sum())} | rule-rows RMSE Fc {rm(te.y.to_numpy()[s]-te.Fc.to_numpy()[s]) if s.any() else 0:.0f} -> {rm(te.y.to_numpy()[s]-P[s]) if s.any() else 0:.0f} | month {np.sqrt(a_.mean()):.3f} -> {np.sqrt(b_.mean()):.3f}")
    A,B=np.concatenate(sA),np.concatenate(sB); jj=np.concatenate([np.full(len(x),i<2) for i,x in enumerate(sA)])
    print(f"  pooled {np.sqrt(A.mean()):.3f} -> {np.sqrt(B.mean()):.3f} ({np.sqrt(B.mean())-np.sqrt(A.mean()):+.3f}) | Jan+Jul {np.sqrt(A[jj].mean()):.3f} -> {np.sqrt(B[jj].mean()):.3f} ({np.sqrt(B[jj].mean())-np.sqrt(A[jj].mean()):+.3f})")
rk=cio.load_ranking(columns=["MVT_ID_mvt","PHASE_mvt","ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","FLIGHT_ID_mvt"]); rk=rk[rk.PHASE_mvt.eq("DEP")]
ms=(pd.to_datetime(rk.MVT_TIME_UTC_mvt,utc=True)-pd.to_datetime(rk.SCHED_TIME_UTC_mvt,utc=True)).dt.total_seconds(); nm=rk.FLIGHT_ID_mvt.isna()
print("ranking rows: R1", int((nm&(ms>6*3600)).sum()), "| R2", int((nm&rk.ADEP_mvt.eq("LIRF")&(ms>15*3600)&(ms<=30*3600)).sum()))
# robustness of R1: drop the k best rows, per-airport lines on all 7 months
print("== R1 robustness")
sA=[];sB=[];aps=[]
for m in ["01","07","08","09","10","11","12"]:
    tr,te=d[d.mon!=m],d[d.mon==m]; P=te.Fc.to_numpy().copy(); s=(te.nm&(te.ms>6*3600)).to_numpy(); st=(tr.nm&(tr.ms>6*3600)).to_numpy()
    for a in te.ap[s].unique():
        k=st&tr.ap.eq(a).to_numpy()
        if k.sum()<20: continue
        ab=line(tr.ms.to_numpy()[k],tr.y.to_numpy()[k]); kk=s&te.ap.eq(a).to_numpy(); P[kk]=ab[0]+ab[1]*te.ms.to_numpy()[kk]
    P=lobt_clip(P,te.mv.to_numpy(),te.lb.to_numpy())
    sA.append((te.y-te.Fc).to_numpy()**2); sB.append((te.y.to_numpy()-P)**2); aps.append(te.ap.to_numpy())
A,B,AP=np.concatenate(sA),np.concatenate(sB),np.concatenate(aps); g=A-B; o=np.argsort(-g)
for k in (0,5,10,20):
    keep=np.ones(len(g),bool); keep[o[:k]]=False; print(f"  drop best {k}: {np.sqrt(A[keep].mean()):.3f} -> {np.sqrt(B[keep].mean()):.3f} ({np.sqrt(B[keep].mean())-np.sqrt(A[keep].mean()):+.3f})")
print("  rows changed", int((g!=0).sum()), "| rows better", int((g>0).sum()), "| worse", int((g<0).sum()))
print("  gain by airport (share):", {a: round(float(g[AP==a].sum()/g.sum()),2) for a in np.unique(AP) if g[AP==a].sum()!=0})
for a in sorted(d.ap.unique()):
    k=(d.nm&(d.ms>6*3600)&d.ap.eq(a)).to_numpy()
    if k.sum()>=20: ab=line(d.ms.to_numpy()[k],d.y.to_numpy()[k]); print(f"  line {a}: n {int(k.sum())} a {ab[0]:.0f} b {ab[1]:.3f}")
