# SUBMISSION — which model to upload

> **Current summary: [`WRITEUP.md`](WRITEUP.md)** — best board score v31 = 289.7783 (2026-09-24).


> **SUPERSEDED — this records only v1-v3 from the first round.** The build
> went on to submit through **v27**; the final, shipped submission is **v25
> (RMSE 295.21)** — a per-airport blend of `per_airport` + `airport_operator_blend`
> (LightGBM) with an 18-seed-averaged MLP, refit on the full 12-month period.
> See [`PIPELINE_REVIEW.md`](PIPELINE_REVIEW.md) §"Bottom line — what to ship"
> and [`plans/13_live_board_next_steps.md`](plans/13_live_board_next_steps.md)
> for the real submission-by-submission history and board scores.

Status of the build: **`STATUS: in progress`** (see [RETROSPECTIVE.md](RETROSPECTIVE.md)).
This file is the decision record for the leaderboard upload only. Nothing here is
written back to `INDEX_MODE2.md`, `CASES.md` or `methodology/`.

---

## Bottom line

| | file | arm / config | use it as |
|---|---|---|---|
| **Primary** | `submissions/<team>_v1.parquet` | `ensemble_antigravity`, untagged | **the protocol pick** — val‑selected winner, register this one |
| Bet | `submissions/<team>_v3.parquet` | `per_airport_stack`, `tag=inner2` | **the leaderboard bet** — better model of the actual process; best in‑band RMSE of the three |
| Safe | `submissions/<team>_v2.parquet` | `lightgbm`, `tag=lr05_lv511_md50` | **the low‑variance fallback** — test score barely moves across seeds |

**If you upload one file:** `v1`. It is the winner selected on the validation
surface (the only honest selection surface — see PLAN.md §4), which is what the
workspace protocol says to submit and register.

**If the leaderboard lets you keep the best of several:** upload all three. They
exist specifically to let the leaderboard resolve a disagreement the held‑out
test set cannot (RETROSPECTIVE.md §4c / §4f): `v1` wins the pooled metric **only
if** a date‑artifact row lands in the scored 215,876 (expected count ~1–2), and
loses ~11 s of decidable RMSE if none does; `v3` is the better model on every row
the metric can actually decide; `v2` is the hedge whose score is stable to ~2 s.

---

## Read this before trusting any number below

1. **All three shipped files are `--split full`** — a refit on all 12 months of
   2025 with hyperparameters frozen from the `forward` run. `full` has **no
   validation or test part** (PLAN.md §4), so **none of the shipped files has a
   score of its own.** Every RMSE in this document is the score of the *same
   arm/config* trained on the `forward` split (train 2025‑01..08 / val 09..10 /
   test 11..12). The shipped file is that model refit on more data; it is
   expected to be at least as good, but that is an assumption, not a measurement.
2. **The files on disk are named `PLACEHOLDER_TEAM_v*`.** The ranking system
   rejects any file whose name is not `<team-name>_v<int>.parquet` with the team
   name as registered with the PRC (`submit.py` docstring). **They must be
   rebuilt with `--team <real-name>` before upload** — commands below.
3. Numbers are seed 42, `with_aobt`, `keep_all` outlier policy, prediction cap
   86,400 s, feature fingerprint `47cba1fb5f800c8d` (identical across all three).

---

## The three files

Forward‑split scores of each shipped config. "pooled" = the leaderboard metric.
"in‑band" = RMSE over rows with target ≤ 7,200 s (the tail‑excluded companion
metric). "decidable" = RMSE with the 1 val / 3 test date‑artifact rows removed
(RETROSPECTIVE.md §4a); it is only on disk for some arms, noted per row.

### v1 — `ensemble_antigravity` (untagged) — PRIMARY / protocol pick

| part | pooled RMSE | in‑band RMSE | decidable RMSE |
|---|---|---|---|
| forward/val | **254.40** | 236.10 | 244.68 |
| forward/test | **270.36** | 251.62 | 255.25 |

- Blend weights (from the shipped `full` run): `mlp_antigravity` 0.886,
  `lightgbm` 0.114, all others 0. So this is ~7/8 the MLP arm.
- **Val‑selected winner.** Val ranking is identical at seeds 42/43/44 and the
  ~45 s val margin over LightGBM is ~7× the largest val seed range
  (RETROSPECTIVE.md §4c) — a legitimate selection, not single‑seed noise.
- **Its test score is a lottery.** Seed range on test is **270.4 – 306.0**
  (val range only 6.5). The variance comes entirely from the LIRF /
  no‑flight‑record cell where the neural arms bid 31k–84k s and LightGBM bids
  20k–26k, so one artifact row landing in the scored part swings the score tens
  of seconds. Registered number is the seed‑42 value; the range is part of the
  result.
- **On rows the metric can decide it is 6th** of the arms built
  (RETROSPECTIVE.md §4f). Its whole pooled lead is three rows.

### v3 — `per_airport_stack` `tag=inner2` — the leaderboard bet

| part | pooled RMSE | in‑band RMSE | decidable RMSE |
|---|---|---|---|
| forward/val | 281.93 | **240.30** | 264.78 * |
| forward/test | 286.44 | **223.58** | 231.25 * |

\* decidable is on disk only for the `inner1` sibling (`artifacts/triage/bands__forward-with_aobt-keep_all__seed42.json`);
the shipped `inner2` config was not run through `stage8_triage.py`. `inner1` and
`inner2` differ only in the inner‑holdout length used to fit the per‑airport
blend weights, and RETROSPECTIVE.md §4f treats the two as one family. Its
`in‑band` RMSE (223.58 test) **is** on disk for the shipped config and is the
best of the three files by ~28 s.

- Per‑airport LightGBM (own stopping point per airport, 72–195 rounds) blended
  with the pooled model, blend weight per airport learned on a 2‑month inner
  holdout: pure per‑airport at LIRF and LSZH, blended elsewhere.
- Real bulk gain, not tail luck: in‑band improves too (RETROSPECTIVE.md §4f).
  Most seed‑stable of the arms that gained (`per_airport` val seed range 0.29;
  the stack wins 3/3 paired seeds).
- Loses the *pooled* metric to `v1` on `forward/test` because of the 3 artifact
  rows; wins on everything else.

### v2 — `lightgbm` `tag=lr05_lv511_md50` — the safe fallback

| part | pooled RMSE | in‑band RMSE | decidable RMSE |
|---|---|---|---|
| forward/val | 295.94 | 253.88 | — (not run through triage) |
| forward/test | *not measured* | *not measured* | *not measured* |

- The LightGBM tuning sweep (RETROSPECTIVE.md §4d) reported **val only**; no
  `eval_test.json` exists for this tag. For scale, the *untuned* base LightGBM
  (`base_lr05_lv255_md100`) scored test pooled **310.63** / decidable **243.77**
  (RETROSPECTIVE.md §4c/§4a), and tuning moved val by only 6.5 s, so the tuned
  test number is expected to be close to those.
- Early stopping fired (best iter 153 of 5000 cap); converged.
- **Why it is the fallback:** LightGBM's test seed range is **2.29 s** vs the
  ensemble's 35.6 s (RETROSPECTIVE.md §4c). If you do not want to bet on tail
  composition, this is the file whose leaderboard score you can predict.

---

## Rebuild with the real team name, then upload

`submit.py` does **not** upload — it writes the file and prints the `mc` command.
Run each command from `Mode2Test/solution/`. Replace `TEAM` with the name
registered with the PRC. Keep `--version` as shown unless the PRC name already
has submissions, in which case start `--version` past the last used integer.

```bash
# v1 — ensemble_antigravity, protocol pick
python3 submit.py --arm ensemble_antigravity --split full --featureset with_aobt \
  --outlier-policy keep_all --seed 42 --team TEAM --version 1

# v2 — tuned LightGBM, low-variance fallback
python3 submit.py --arm lightgbm --tag lr05_lv511_md50 --split full --featureset with_aobt \
  --outlier-policy keep_all --seed 42 --team TEAM --version 2

# v3 — per_airport_stack (inner=2mo), leaderboard bet
python3 submit.py --arm per_airport_stack --tag inner2 --split full --featureset with_aobt \
  --outlier-policy keep_all --seed 42 --team TEAM --version 3
```

Each run rewrites `submissions/TEAM_v<n>.parquet` and a `.meta.json` provenance
block (every contributing run's path, feature fingerprint, round count, params).
Then run the printed `mc cp` command yourself to upload.

---

## Open items — resolve against the challenge rules before uploading

- **Does the leaderboard keep the best submission or the latest?** And is there a
  slot cap or a per‑day limit? The "upload all three" plan only makes sense if
  best‑of‑submissions counts. Check `ansperformance.eu/study/data-challenge/dc2026/ranking.html`
  (linked in `submit.py`).
- **`--version` numbering** must be a fresh incremental integer under the real
  team name — confirm nothing has been submitted under it yet.
- **`AOBT_3` feed on competition day.** All three use `with_aobt`. If that feed
  fails at scoring time the cost is ~4 s RMSE, not the ~160 s the standalone
  proxy figure suggests (RETROSPECTIVE.md §4b) — not worth a `no_aobt` rebuild.

---

## Provenance

Per‑file provenance blocks: `submissions/*.parquet.meta.json`.
Full reasoning for the three‑file strategy: [RETROSPECTIVE.md](RETROSPECTIVE.md)
§4a (the date‑artifact rows), §4c (seed sweep), §4d (LightGBM tuning), §4f
(per‑airport family), §4e (closing‑gate fixes to submission provenance).
