"""Shared configuration for the PRC Data Challenge 2026 taxi-out build.

One place for every path, column name and contract constant, so no pipeline
hardcodes a path or silently disagrees with a sibling about what a split is.
"""
from __future__ import annotations

from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parent
DATA_DIR = SOLUTION_DIR.parent / "prc-2026-datasets"
ARTIFACTS = SOLUTION_DIR / "artifacts"
SPLIT_DIR = ARTIFACTS / "splits"
FEATURE_DIR = ARTIFACTS / "features"
AUDIT_DIR = ARTIFACTS / "audit"

RANKING_PARQUET = DATA_DIR / "ranking.parquet"
SUBMITTING_PARQUET = DATA_DIR / "submitting.parquet"

TARGET = "TAXITIME_SEC_mvt"
ID = "MVT_ID_mvt"
METRIC = "rmse"          # lower is better -- every ranking helper must respect this
HIGHER_IS_BETTER = False

# ---------------------------------------------------------------------------
# The ranking-availability contract.
#
# ranking.parquet blanks exactly these two columns on DEP rows. Any feature a
# model consumes MUST be derivable without them, or the model cannot be scored.
# Verified empirically by stage1_audit.py (do not edit from memory -- the audit
# asserts this list against the real file).
# ---------------------------------------------------------------------------
BLANKED_ON_RANKING_DEP = ("BLOCK_TIME_UTC_mvt", TARGET)

# Columns that are fully available for DEP rows at prediction time.
DEP_AVAILABLE = (
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt",
    "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
    "LOBT_flt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
)

# The near-target proxy the organisers left in the scored set. Isolated here so
# the no_aobt ablation is one constant, not a grep across three pipelines.
PROXY_COLS = ("AOBT_3_flt",)

# ---------------------------------------------------------------------------
# Target handling. THREE separate quantities that a first draft of this file
# conflated into two constants -- design-gate finding B1/"decouple the clip".
#
# 1. IMPOSSIBLE-VALUE FLOOR. A taxi-out of <= 0 s cannot happen; 388 training
#    rows have one (min -12 s). These are dropped from training unconditionally:
#    they are corrupt, not tail.
# 2. OUTLIER POLICY (swept, not hardcoded). What to do with the heavy upper tail
#    in the TRAINING pool. The first draft dropped everything above 7200 s and
#    called it costless. It is not: 82.4% of those rows have no flight record,
#    a subpopulation `has_flight_record` already identifies, so dropping them
#    removes learnable structure and biases every affected cell low under L2.
#    Measured on forward/val with an ADEP x has_flight_record group-mean model:
#    unfiltered 445.85 / winsorised 448.30 / dropped 454.68. The policy is
#    therefore an axis to sweep, and `drop` is expected to lose.
# 3. PREDICTION CLIP. A safety net on model OUTPUT, mainly for the ridge arm's
#    linear extrapolation. Conceptually unrelated to (2): it must sit high
#    enough not to muzzle a model that legitimately predicts a long taxi.
#    p99.99 of the training target is 10,712 s, so that is the cap.
#
# Evaluation sets are NEVER filtered under any policy -- the leaderboard scores
# every row, and an evaluation set filtered to make a number look better is a
# number for an easier problem than the one posed.
# ---------------------------------------------------------------------------
TARGET_IMPOSSIBLE_MAX_SEC = 0          # y <= this is corrupt; always dropped
TRAIN_TARGET_MIN_SEC = 60              # lower bound for the `drop` policy only

OUTLIER_POLICIES = {
    # name           -> (mode, bound_seconds)
    "keep_all":       ("keep", None),
    "winsor_7200":    ("winsorize", 7200),
    "winsor_p9999":   ("winsorize", 10712),   # p99.99 of the 2025 training target
    "winsor_21600":   ("winsorize", 21600),
    # Added at the closing gate, from the band decomposition in
    # stage6_clip_sweep.py: the training target has three regimes, not two. Below
    # 7,200 s is the bulk; 7,200-80,000 s is a real, learnable heavy tail (92 val
    # rows carrying 16% of val squared error, and keep_all halves their RMSE
    # against every winsorising policy); above 80,000 s are exactly 15 rows in
    # 2.085M whose off-block timestamp is one calendar day before takeoff. Every
    # bound below 21,600 s cuts into the LEARNABLE tail to reach the artifacts.
    # These two cut between the regimes instead: they leave the learnable tail
    # untouched and cap only the 15 rows whose L2 gradient is ~80,000 s wide.
    "winsor_43200":   ("winsorize", 43200),   # 12 h
    "winsor_80000":   ("winsorize", 80000),   # exactly the artifact boundary
    "drop_7200":      ("drop", 7200),         # the original, expected to lose
}
# Changed from winsor_p9999 after the policy sweep was re-run at one code version
# (the first sweep compared 49-feature runs against a 57-feature one and was not a
# clean comparison). Decomposed on val -- 369,493 in-band rows, 92 mid-tail rows
# (7200 < y < 80000) and 1 date-artifact row -- keep_all is better because it
# halves the mid-tail RMSE (15,014 -> 7,675 s), worth ~1.5e10 of squared error,
# against an in-band cost of ~0.4e10. Re-ranked with the single date-artifact val
# row EXCLUDED (`decidable_rmse`), because raising the cap moves that one row's
# error by ~2.3e9 and could have carried the decision on its own: the pooled and
# decidable top-5 orderings are identical, so the choice does not rest on it
# (artifacts/clip/policy_cap_decision.json). That is the correct trade for a POOLED
# squared-error metric: under L2 the optimal prediction is the conditional mean,
# and every winsorising policy biases that mean downward wherever the tail has
# mass. The in-band RMSE rises (230.9 -> 252.8) precisely because the predictions
# stop being biased low; it is a price, not a regression.
DEFAULT_OUTLIER_POLICY = "keep_all"

PREDICTION_CLIP_MIN_SEC = 60
# Raised from 10,712 s (p99.99 of the training target) after stage6_clip_sweep.py
# measured what the cap actually costs. The cap is a HYPERPARAMETER, not a fact
# about the data, and at 10,712 s it was truncating predictions the model was
# right to make: under keep_all the val RMSE runs 337.82 / 326.27 / 305.29 /
# 302.45 at caps of 7200 / 10712 / 21600 / none. 86400 s (one day) is chosen
# because it never binds in practice -- every cap at or above 43,200 s scores
# identically -- while still refusing a physically absurd emission, and because
# one day is exactly where the target's date-artifact regime begins (see BANDS
# in stage6_clip_sweep.py). Selected on VAL. The repeat-seed check has now RUN
# (`--check-seeds 43,44`, once keep_all models existed at those seeds) and all
# three seeds agree: best cap 43,200 s, with 43,200 / 86,400 / no-cap scoring
# identically (302.45) because only one val row sits above 43,200. So the cap is
# stable, and anything at or above 43,200 s is the same decision -- 86,400 s is
# chosen among the ties for the reason above, not because it measured better.
#
# Consequence to state rather than discover later: with the cap this high the
# arithmetic hard floor of design-gate B3 (`common/evaluate.hard_floor_rmse`) is
# ~0 on every part, because almost no row exceeds the cap. That floor was
# load-bearing in the believability band and no longer discriminates anything;
# the band decomposition in stage6_clip_sweep.py replaces it as the instrument
# for reading a suspiciously good number.
PREDICTION_CLIP_MAX_SEC = 86400

# Retained so a reader of an older artifact can still resolve the name; the
# training pool no longer uses it as a filter bound.
TRAIN_TARGET_MAX_SEC = 7200

# ---------------------------------------------------------------------------
# Split protocol (see PLAN.md sec.4). Months are 'YYYY-MM' of MVT_TIME_UTC_mvt.
# 'forward' is the PRIMARY honest split: strictly time-ordered, so val and test
# are in the future relative to train, mirroring train-2025 -> score-2026.
# 'seasonal' is the SECONDARY composition-matched probe: it holds out exactly the
# months the leaderboard scores (Jan + Jul), with Jul restricted to the three
# airports the ranking file actually covers in Jul 2026.
# ---------------------------------------------------------------------------
SPLITS = {
    "forward": {
        "train": ["2025-01", "2025-02", "2025-03", "2025-04",
                  "2025-05", "2025-06", "2025-07", "2025-08"],
        "val":   ["2025-09", "2025-10"],
        "test":  ["2025-11", "2025-12"],
        "airport_filter": None,
    },
    "seasonal": {
        "train": ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
                  "2025-08", "2025-09", "2025-10", "2025-11"],
        "val":   ["2025-12"],
        "test":  ["2025-01", "2025-07"],
        # Jul 2026 in ranking.parquet covers only these three airports.
        "airport_filter": {"2025-07": ["EDDF", "EGLL", "EHAM"]},
    },
}
# The submission model. Design-gate finding B4: the plan said "refit on all 12
# months with hyperparameters frozen from the honest split" and nothing
# implemented it -- submit.py would have shipped the Jan-Aug model, discarding
# Sep-Dec, the four months CLOSEST to the scored January on a task the audit
# itself shows is drifting. This pseudo-split makes the refit a real code path.
# It has no val/test: nothing may be selected on it, only fitted. Every
# hyperparameter must arrive frozen from a `forward` run.
SPLITS["full"] = {
    "train": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
              "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"],
    "val": [], "test": [], "airport_filter": None,
}
# The INNER selection surface, for stacking weights that must not be fitted on
# val. It is `forward` shifted back two months: train Jan-Jun, "val" Jul-Aug.
# Arms trained here produce predictions on Jul-Aug that no `forward` model has
# seen, so a blend weight solved on them is genuinely out-of-sample -- which is
# what separates a stack from an arm that fits its parameters on the validation
# labels (`VAL_FITTED_ARMS`).
#
# Time-based, not random, for the same reason `forward` is: the real task
# extrapolates forward, and a randomly-carved inner fold would let a weight be
# tuned using the future.
#
# It has no `test`: nothing may be *evaluated* here, only fitted. And it is
# selection-forbidden, so compare.py will not crown an arm on it -- it is an
# auxiliary surface, not a second place to pick a winner.
SPLITS["forward_inner"] = {
    "train": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
    "val":   ["2025-07", "2025-08"],
    "test":  [],
    "airport_filter": None,
}

SELECTION_FORBIDDEN_SPLITS = ("full", "seasonal", "forward_inner")

# The candidate arms this concise build actually has. early_baseline/ grew this list
# to 18+ pipelines (ridge, xgboost_antigravity, cross_blend, per_airport_*, ...)
# as it iterated past the two floor arms; none of those pipeline_*/ dirs exist
# here, so this list is scoped to match rather than copied verbatim.
ARMS = ("naive_offblock", "lightgbm")

# No arm here fits free parameters on the validation labels (that only applies
# to early_baseline/'s ensemble_antigravity and cross_blend, neither present in this
# build), so there is nothing to flag as val-fitted.
VAL_FITTED_ARMS = ()

DEFAULT_SPLIT = "forward"

FEATURESETS = ("with_aobt", "no_aobt")
DEFAULT_FEATURESET = "with_aobt"

SEED = 42
SWEEP_SEEDS = (42, 43, 44)


def protocol_key(split: str, featureset: str, outlier_policy: str = None) -> str:
    """The single artifact-path key. run_dir(arm, protocol=..., seed=...) uses it.

    The outlier policy is part of the key: two runs differing only in how the
    training tail was handled are different experiments, and a shared path would
    let one silently overwrite the other (MISTAKES.md #35/#38).
    """
    pol = DEFAULT_OUTLIER_POLICY if outlier_policy is None else outlier_policy
    if pol not in OUTLIER_POLICIES:
        raise ValueError("unknown outlier policy {!r}; known: {}".format(
            pol, sorted(OUTLIER_POLICIES)))
    return "{}-{}-{}".format(split, featureset, pol)


def feature_code_fingerprint() -> str:
    """Hash of the code + constants that determine a feature cache's contents.

    Design-gate finding S6: the cache had no invalidation, so any edit to
    features.py would have been silently ignored on the next run -- the worst
    kind of staleness, because everything still executes and reports success.
    """
    import hashlib
    h = hashlib.sha256()
    for rel in ("common/features.py", "common/io.py"):
        h.update((SOLUTION_DIR / rel).read_bytes())
    h.update(repr(sorted(PROXY_COLS)).encode())
    h.update(repr(sorted(BLANKED_ON_RANKING_DEP)).encode())
    return h.hexdigest()[:16]
