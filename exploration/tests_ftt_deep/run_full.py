"""Train the deep FTT `full` seeds with epochs = ceil(median forward best) of the deep forward seeds."""
import math, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
# Amendment 2: forward stage skipped; use the forward_inner best epochs.
b = [int(np.load(ROOT / "tests_ftt" / f"preds_seed{s}_forward_inner_d96b6.npz")["best_epoch"]) for s in (42, 43, 44)]
E = math.ceil(float(np.median(b)))
print(f"forward_inner best epochs {b} -> full freeze epochs {E}", flush=True)
sys.exit(subprocess.call([sys.executable, "-u", str(ROOT / "tests_ftt" / "run_ftt.py"), "42", "43", "44",
                          "--lr", "5e-4", "--d-token", "96", "--n-blocks", "6", "--amp", "--tag", "d96b6",
                          "--split", "full", "--freeze-epochs", str(E), "--predict-ranking"]))
