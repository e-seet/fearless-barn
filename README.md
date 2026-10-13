# PRC Data Challenge 2026 — taxi-out time prediction

Team **fearless-barn**. Predicts `TAXITIME_SEC_mvt` (seconds from off-block to takeoff) for departures
at ten European airports, for the EUROCONTROL PRC Data Challenge 2026.

| | |
|---|---|
| **Final round file** | [`exploration/submissions/fearless-barn_final.parquet`](exploration/submissions/fearless-barn_final.parquet): 670,790 departures (Jan, Feb, Jun, Jul 2026) |
| **Best public board score** | **269.9888** pooled RMSE (January + July 2026) |
| **Same base model without ADS-B** | 286.5256 |
| **Licence** | GNU GPLv3, see [`LICENSE`](LICENSE) |

## Start here

| Document | What it covers |
|---|---|
| **[`exploration/WRITEUP.md`](exploration/WRITEUP.md)** | **The method.** The model stage by stage, how each decision was made, use of the leaderboard, the final round, results, what did not work, reproducibility, data licences, credits |
| [`exploration/EXPLORATION_SUMMARY.md`](exploration/EXPLORATION_SUMMARY.md) | One page: everything tried, with its outcome |
| [`exploration/final_pipeline.py`](exploration/final_pipeline.py) | The whole recipe in one script |

## The model in brief

1. **Base model**, from the competition's data only: a per-airport blend of LightGBM trees, MLPs,
   FT-Transformers and CatBoost, with TabNet at Rome.
2. **ADS-B corrector** at seven airports: a LightGBM model on the base's error, fed by each
   departure's own ground track from the open adsb.lol archive.
3. **Weather offsets** from METAR: frost in January, thunderstorms in July.
4. **Rome and Paris CDG correctors**, also from ADS-B ground tracks.
5. **LOBT window**: predictions are clipped to the range the flight plan allows.

Every design decision was tested against a pass bar written down before the result existed. The
write-up lists what was adopted, what was rejected, and why.

## Run it

```bash
cd exploration
pip install -r requirements.txt
python final_pipeline.py --movements <movements.parquet> --points-dir <adsb_dir> --fetch --out <pred.parquet>
```

The movements file must have `ranking.parquet`'s schema. `--fetch` downloads the ADS-B archives and
METAR reports it needs. Stage 1 needs the trained base models (about 37 GB), which are too large for
git; see write-up §8 for what is in the repository and how the uploaded files were verified.

## Repository layout

| Path | Contents |
|---|---|
| [`exploration/`](exploration/) | The solution: pipeline, training code, every experiment (`tests_*/`, each with its pre-registration and result), submissions. Mapped in write-up §12 |
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Exploratory analysis of the data |
| [`solution/`](solution/) | An early baseline, superseded by `exploration/` and kept for its leakage and split audit |
| `tools/ml_common/` | Shared checks the training code imports (seeding, split disjointness, leak controls) |

## Data

- **Competition data** is not redistributed here. It comes from the organizers' bucket and goes in
  `prc-2026-datasets/` (git-ignored).
- **External data** is open and listed with its licences in write-up §10: adsb.lol ADS-B (ODbL) and
  METAR from the Iowa Environmental Mesonet.

## Licence

Released under the **GNU General Public License v3.0**; see [`LICENSE`](LICENSE).
