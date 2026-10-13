# Do not upload

v51–v54 are weather-offset variants built by another agent on 2026-10-01 and never uploaded.

- **Per-airport frost offsets are not supported.** Their per-airport frost offsets failed an
  out-of-sample check (`tests_weather_validated/RESULT.md`).
- **Uploading them would fit the hidden labels.** Uploading several and keeping the best would
  select on the hidden 2026 labels.
- **They would replace the final submission.** Ranking uses the **last** upload, so uploading any
  of these would replace v50.

Kept only for the record.
