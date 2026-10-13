# PRC Data Challenge 2026 — taxi-out time prediction

Team **fearless-barn**. Predicts `TAXITIME_SEC_mvt` (seconds from off-block to takeoff) for departures
at ten European airports, for the EUROCONTROL PRC Data Challenge 2026.

| | |
|---|---|
| **Final round file** | [`solution/submissions/fearless-barn_final.parquet`](solution/submissions/fearless-barn_final.parquet): 670,790 departures (Jan, Feb, Jun, Jul 2026) |
| **Public leaderboard score** | **269.9888** RMSE. The leaderboard scores January + July 2026 only (344,841 departures, all airports pooled) |
| **Final round score** | none yet: the final ranking is blind |
| **Same base model without ADS-B** | 286.5256 |
| **Licence** | GNU GPLv3, see [`LICENSE`](LICENSE) |

## Start here

| Document | What it covers |
|---|---|
| **[`solution/WRITEUP.md`](solution/WRITEUP.md)** | **The method.** The model stage by stage, how each decision was made, use of the leaderboard, the final round, results, what did not work, reproducibility, data licences, credits |
| [`exploration/EXPLORATION_SUMMARY.md`](exploration/EXPLORATION_SUMMARY.md) | One page: everything tried, with its outcome |
| [`solution/final_pipeline.py`](solution/final_pipeline.py) | The whole recipe in one script |

## The model in brief

A base model predicts every departure from the competition's own records. It can't see what happened
on the apron, so for the departures whose aircraft appears in an open ADS-B archive, a second model
reads the aircraft's own ground track and corrects the prediction. A few narrow rules cover what
neither model sees.

| Stage | What it does | Public leaderboard after adding it |
|---|---|---:|
| 1. Base model | Per-airport blend of LightGBM trees, MLPs, FT-Transformers and CatBoost, with TabNet at Rome. Competition data only | 286.53 |
| 2. ADS-B corrector | A LightGBM model on the base model's error, at seven airports, from each departure's own ground track (adsb.lol) | 272.17 |
| 3. Weather offsets | Frost in January, thunderstorms in July, from METAR | 271.87 |
| 4–5. Rome and Paris CDG correctors | The same idea, one airport each | 271.56 |
| Later refinements | A larger corrector, a wider base blend, a better training base | **269.99** |
| 6. LOBT window | Predictions clipped to the range the flight plan allows; added for the final round | not scored |

**How decisions were made.** Every change was tested on held-out months of 2025 against a pass bar
written down before the result existed. Changes that looked good but rested on a handful of extreme
flights were rejected, including some that would probably have scored better. The write-up lists what
was adopted, what was rejected, and why, and states exactly how the public leaderboard was used.

## Run it

```bash
cd solution
pip install -r requirements.txt
python final_pipeline.py --movements <movements.parquet> --points-dir <adsb_dir> --fetch --out <pred.parquet>
```

The movements file must have `ranking.parquet`'s schema. `--fetch` downloads the ADS-B archives and
METAR reports it needs. Stage 1 needs the trained base models (about 37 GB), which are too large for
git; see write-up §8 for what is in the repository and how the uploaded files were verified.

## Repository layout

| Path | Contents |
|---|---|
| [`solution/`](solution/) | **The final solution:** the pipeline, the code that trains its models, the saved models and files it reads, the submissions. Mapped in [`solution/README.md`](solution/README.md) |
| [`exploration/`](exploration/) | **The experiments behind it:** 58 experiments (each with its pre-registration and result), the candidate models that were not selected, early analysis scripts, decision history. Indexed in [`exploration/README.md`](exploration/README.md) |
| `tools/` | Shared checks the training code imports (`ml_common/`), and the path helper the experiment scripts use (`repo_layout.py`) |

## Data

- **Competition data** is not redistributed here. It comes from the organizers' bucket and goes in
  `prc-2026-datasets/` (git-ignored).
- **External data** is open and listed with its licences in write-up §10: adsb.lol ADS-B (ODbL) and
  METAR from the Iowa Environmental Mesonet.

## Licence

Released under the **GNU General Public License v3.0**; see [`LICENSE`](LICENSE).
