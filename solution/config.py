"""Shared configuration for the PRC Data Challenge 2026 taxi-out build.

One place for every path, column name and contract constant, so no pipeline
hardcodes a path or silently disagrees with a sibling about what a split is.
"""
from __future__ import annotations

import sys
from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parent
DATA_DIR = SOLUTION_DIR.parent / "prc-2026-datasets"

# ---------------------------------------------------------------------------
# Quarantine for arms the pipeline review judged OVER-TUNED (see
# overtuned/README.md and PIPELINE_REVIEW.md tier E). They are removed from
# ARMS, so run_all.py never builds them and compare.py never ranks them.
#
# Their CODE stays importable from here rather than being deleted, for two
# reasons: (1) the runs already on disk under artifacts/runs/ are the EVIDENCE
# for the over-tuning verdict, and a verdict whose evidence cannot be
# regenerated is not reviewable; (2) `airport_lirf_identity_multimodel` (tier
# A/B, still a live arm) predicts through airport_multiseed_multimodel ->
# airport_unsup_multimodel, so common/predict.py must still resolve
# `pipeline_airport_unsup_multimodel.train`. One sys.path entry keeps every
# existing import site working unchanged; the folder move is what marks them.
# ---------------------------------------------------------------------------
OVERTUNED_DIR = SOLUTION_DIR / "overtuned"
if OVERTUNED_DIR.is_dir() and str(OVERTUNED_DIR) not in sys.path:
    sys.path.append(str(OVERTUNED_DIR))

ARTIFACTS = SOLUTION_DIR / "artifacts"
SPLIT_DIR = ARTIFACTS / "splits"
FEATURE_DIR = ARTIFACTS / "features"
AUDIT_DIR = ARTIFACTS / "audit"

RANKING_PARQUET = DATA_DIR / "ranking.parquet"
SUBMITTING_PARQUET = DATA_DIR / "submitting.parquet"

# Team name, as registered with the PRC. One constant, because submit.py's
# --team was a required free-text argument with no default and every submission
# up to v13 was therefore typed by hand as "PLACEHOLDER_TEAM" -- a placeholder
# that made it into 12 built submission filenames. The competition contract keys
# the file on this name (<team>_v<N>.parquet), so it is a data constant, not a
# preference.
TEAM = "fearless_barn"

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
# months the leaderboard scores (Jan + Jul).
#
# STALE ASSUMPTION, KEPT FOR ARTIFACT COMPATIBILITY (found 2026-09-17,
# TAXI_OUT_DRIVERS.md sec.2af): 'seasonal's airport_filter below restricts Jul
# to three airports on the belief that "Jul 2026 in ranking.parquet covers
# only these three airports" -- checked directly against the real
# ranking.parquet on disk and this is FALSE: all 10 target airports have
# substantial July 2026 row counts (LIRF alone: 15,838 rows). Every
# comparison already run on 'seasonal' stays a valid A-vs-B comparison (the
# filter under-weights, doesn't corrupt, 7 airports in July) but the phrase
# "board-composition-matched" overstates what 'seasonal' actually covers for
# those 7 airports in July. NOT edited in place: artifacts/splits/seasonal/
# and everything built against it stay interpretable as originally reported.
# Use 'seasonal_full' (below) for a check that needs the real July
# composition, e.g. anything specifically about LIRF's July presence.
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
        # STALE, see the module-level note above -- kept as originally
        # written so existing artifacts stay reproducible byte-for-byte.
        "airport_filter": {"2025-07": ["EDDF", "EGLL", "EHAM"]},
    },
    # Same as 'seasonal' but WITHOUT the stale July airport restriction --
    # the actually board-composition-matched version, added 2026-09-17
    # rather than editing 'seasonal' in place (see note above).
    "seasonal_full": {
        "train": ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
                  "2025-08", "2025-09", "2025-10", "2025-11"],
        "val":   ["2025-12"],
        "test":  ["2025-01", "2025-07"],
        "airport_filter": None,
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

# Held-out base predictions for Feb-Jun 2025, so the ADS-B corrector can train on those
# months (tests_febjun/PREREGISTERED.md). Selection-forbidden: it only supplies residuals.
SPLITS["holdout_febjun"] = {
    "train": ["2025-01", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11"],
    "val":   ["2025-12"],
    "test":  ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
    "airport_filter": None,
}

SELECTION_FORBIDDEN_SPLITS = ("full", "seasonal", "seasonal_full", "forward_inner", "holdout_febjun")

# ---------------------------------------------------------------------------
# Arms whose FREE PARAMETERS are fitted directly on the validation labels, so
# their val score is in-sample for those parameters and is NOT comparable, at
# face value, against arms that only early-stopped on val.
#
# Read from pipeline_ensemble_antigravity/train.py: it solves
# `nnls(P, y_val)` for the blend weights over up to five base arms, then reports
# the resulting val RMSE. Early stopping selects ONE integer against a patience
# rule; NNLS fits k continuous weights to minimise val RMSE outright -- a
# different kind of use of the same surface. The test numbers behave exactly as
# that predicts (val 325.39 -> test 355.18, against lightgbm's 330.20 -> 349.02).
#
# This is a caveat, not a disqualification: the arm is ranked and reported, but
# compare.py stamps the summary so no reader takes its val crown as established.
# The honest fix is an inner selection surface carved out of TRAIN, which is what
# pipeline_blend/ (this build's own blend arm) does.
# ---------------------------------------------------------------------------
VAL_FITTED_ARMS = ("ensemble_antigravity", "cross_blend")

# The candidate arms, in floor-to-complex order. ONE list: run_all.py, compare.py
# and submit.py each kept their own, and they drifted -- submit.py's argparse
# `choices` still listed only the three original arms, so the val-selected winner
# could not be submitted at all while `predict_ranking` supported it fine. A
# per-file copy of a list that must agree is the same defect shape as a per-build
# copy of a check (root CLAUDE.md on tools/ml_common).
ARMS = ("naive_offblock", "ridge", "lightgbm",
        "xgboost_antigravity", "mlp_antigravity", "ensemble_antigravity",
        "delay_twostage_antigravity", "per_airport_antigravity", "feature_pruned_antigravity",
        "multimodel_antigravity",
        # Added in the improvement round. per_airport tests whether segmenting by
        # departure airport beats letting the trees split on it; lightgbm_recency
        # tests whether down-weighting older months helps a model that has to
        # extrapolate 13 months forward.
        "per_airport", "lightgbm_recency",
        # Second improvement round: per_airport_stack blends the pooled and
        # per-airport models with weights solved on an inner holdout carved from
        # TRAIN (the Santander / facial-keypoints "stack, don't replace" lever);
        # lightgbm_unsup adds SVD stand/runway embeddings and k-means day regimes,
        # both learned without the target.
        "per_airport_stack", "lightgbm_unsup",
        # Third round: a convex blend of the per-airport and neural families,
        # whose band decompositions are complementary (per-airport wins the
        # 327,907 in-band rows, the neural arms win the 61 tail rows).
        "cross_blend",
        # Bulk-accuracy round: after cross_blend, 82.8% of the remaining test
        # error is the 327,948 in-band rows, so the constraint is now bulk
        # accuracy rather than the tail. lightgbm_arrfeat adds the two channels
        # the base feature set under-uses -- arrival taxi-in structure (observed,
        # and NOT blanked in the scored set) and a runway-specific queue count.
        "lightgbm_arrfeat", "per_airport_arrfeat", "per_airport_tuned",
        # Fourth round: operator had never been tried as a segmentation axis at
        # all (only airport had). per_operator tests it alone; the two segment
        # by different columns on the same rows and turned out complementary
        # enough (an isolated check, not this pipeline: ~271.9 val for a plain
        # 0.5/0.5 blend of their predictions, vs ~278 for either alone) that
        # airport_operator_blend formalises the blend with a weight solved on
        # an inner holdout rather than eyeballed.
        "per_operator", "airport_operator_blend",
        # Fifth round: SAME LEVER as multimodel_antigravity/cross_blend (a
        # per-airport blend of a tree arm and mlp_antigravity), but with
        # airport_operator_blend as the tree-side parent instead of plain
        # per_airport. WON: new best arm in the build, beating
        # multimodel_antigravity at all 3 sweep seeds (val 248.33/248.60/252.42
        # vs 248.81/248.98/262.80; test 246.89/242.62/240.81 vs
        # 247.46/242.99/244.44) -- the margin is largest at seed 44, where
        # mlp_antigravity had its weakest fit: because the per-airport weights
        # lean on airport_operator_blend everywhere except LIRF, this blend is
        # more robust to a bad mlp_antigravity seed than multimodel_antigravity
        # was.
        "airport_operator_multimodel",
        # Verification round: two more arms built to actually test ideas that
        # had only been reasoned about, rather than leave them as guesses.
        # per_airport_flighttype nests an S-vs-Other flight-type split inside
        # per-airport routing -- val 295.77, WORSE than per_airport alone
        # (278.17). The damage is concentrated entirely in LIRF (785 RMSE vs
        # ~530 for other LIRF arms): LIRF's 'Other' cell (2,701 train rows) is
        # too small relative to LIRF's already-high variance, so the split
        # amplifies LIRF's instability instead of specialising usefully.
        # tri_blend solves ONE global NNLS weight over per_airport,
        # per_operator and mlp_antigravity (contrast with the per-airport
        # weighting airport_operator_multimodel uses) -- val 263.13, worse
        # than mlp_antigravity alone (253.99). Confirms directly: the right
        # tree/neural mix genuinely varies by airport (LIRF wants ~90% MLP,
        # every other airport wants ~95%+ tree), so a single global weight
        # cannot express it and just compromises away the MLP's LIRF edge.
        "per_airport_flighttype", "tri_blend",
        # Rounds six to ten built a ladder on top of airport_operator_multimodel
        # -- airport_wtc / airport_runway / airport_wtc_rec / airport_unsup /
        # airport_multiseed / airport_debiased. Each re-slices the SAME per-airport
        # blend weight on a finer bucketing axis (wake-turbulence category, primary
        # runway, record status, k-means congestion clusters), and each was promoted
        # to "champion" on a 3-seed sweep. The 6-seed re-run
        # (tests_seed_robustness/result_seeds_42-47.txt) shows why that promotion
        # was not earned: the whole family's spread is 2.6 s (val) / 4.5 s (test)
        # while a SINGLE member's own seed range is 4.4 s / 13.1 s. The ladder is
        # ranking inside its own noise, and it pays 10 -> ~240 fitted cells plus
        # hand-chosen hyperparameters (K=6, lambda=2000, min_cell_rows walked
        # 1000 -> 300 -> 30) to do it. airport_debiased goes further and tunes
        # thresholds against the reported number outright -- its own docstring
        # justifies gamma=0.50 by "dropping Test RMSE to 226.83 s".
        #
        # They are REMOVED from ARMS and quarantined under overtuned/ (see
        # overtuned/README.md and PIPELINE_REVIEW.md tier E). Their code is still
        # importable via OVERTUNED_DIR above, so the runs on disk stay reproducible.
        #
        # airport_operator_multimodel (above) deliberately STAYS: it is the last
        # honest member of this family -- the same 10 per-airport alphas as
        # multimodel_antigravity, solved on forward_inner, with no bucketing knob
        # and no extra hyperparameter. Only its tree-side parent differs.
        #
        # per_airport_mlp is kept in ARMS as a recorded clean negative: it is not
        # over-tuned, it is non-functional (LIRF val RMSE 1128 s from tail-sample
        # scarcity). Same for per_airport_antigravity, which is byte-identical to
        # per_airport -- duplicated compute, not a tuning problem.
        "per_airport_mlp",
        # QUARANTINE RESTORED 2026-09-21. The three over-tuned arms
        # (airport_unsup_multimodel, airport_multiseed_multimodel,
        # airport_debiased_multimodel) were TEMPORARILY re-enabled here on
        # 2026-09-09 so submit.py --arm would accept them while v4-v12 were
        # regenerated against the refreshed ranking.parquet. That rebuild
        # finished 2026-09-10 and the line was never reverted, so for eleven
        # days submit.py would still accept airport_debiased_multimodel -- the
        # arm whose own docstring justifies its constants by the test score they
        # produce (PIPELINE_REVIEW.md tier E). Reverted now; see
        # CONVERGENCE_AUDIT.md sec.6b.
        #
        # The rebuilt v4-v12 .parquet files stay valid: only submit.py --arm
        # reads this list. Their code stays importable via OVERTUNED_DIR and
        # predict_ranking() keeps its branches, so re-enabling for another
        # refresh is a one-line change -- deliberately one that has to be made
        # on purpose.
        # Zero-parameter LIRF telemetry-identity override, applied ON TOP of a
        # base arm. It was adopted on measured evidence (plan 12, candidate 2)
        # but was never in ARMS, had no predict/submit branch and could not sit
        # on a tagged base -- adoptable but not shippable. Registered here so it
        # can reach a submission like any other arm.
        "airport_lirf_identity_multimodel",
        # Same wall, found 2026-09-21: this arm has a full pipeline, a
        # predict_ranking() branch AND a PARENTS entry in submit.py, but was
        # never in ARMS -- so `submit.py --arm` rejected it even though v26
        # (295.4767) was built from it, and build_submission_index.py has been
        # reporting v26 as NOT REBUILDABLE. Registered now.
        "airport_operator_catboost_multimodel",
        # The TabNet-at-LIRF override behind v29 (293.6685, best board score).
        # Same shape of gap as airport_lirf_identity_multimodel had: it was
        # adopted on measured evidence but built by ad-hoc scripts against a
        # submission parquet, with "not_reproducible_via_submit_py": true in its
        # own meta. Registered so the best score is rebuildable through the
        # normal path and the feature-fingerprint gate can see its chain.
        "tabnet_lirf_override",
        # 3-member per-airport blend adding an FT-Transformer (column attention)
        # to the tree+MLP pair. Confirmed on three independent periods against a
        # bar fixed in advance; honest effect ~1.5s in-band. Registered BEFORE
        # its first submission rather than retrofitted, which is the lesson from
        # v26/v28/v29 (CONVERGENCE_AUDIT.md sec.6a).
        "ftt_multimodel")

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
