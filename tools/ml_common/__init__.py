"""ml_common -- one tested implementation of Mode 2's standing checks.

New Mode 2 builds IMPORT this instead of re-implementing seeding, negative controls,
split gates, triage, baselines, convergence evidence, artifact keying and the
duplicate audit per folder. Already-registered builds are NOT retrofitted.

Import pattern from a solution folder (see tools/ml_common/README.md):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N] / "tools"))
    from ml_common import seed_everything, label_permutation_control, triage

Import-time dependencies are stdlib + numpy ONLY. torch and PIL are imported lazily,
inside the functions that need them, so this package imports on a machine with neither.

Every public function's docstring cites the MISTAKES.md entries it encodes; the
function-to-entry map is in README.md.
"""
from __future__ import annotations

from .artifacts import guard_overwrite, run_dir, run_key, write_json_guarded
from .baselines import measured_chance, stated_floor, trivial_baselines
from .controls import (
    BEST_CHECKPOINT_KEY,
    ControlResult,
    ControlSuite,
    label_permutation_control,
    overfit_one_batch,
    overfit_one_batch_generic,
    select_signal_batch,
)
from .convergence import assert_sweep_converged, convergence_evidence
from .dedup import (
    assert_positive_control_ran,
    connected_components,
    hash_pairs_below,
    md5_file,
    md5_groups,
    reencode_positive_control,
    sign_hash_from_images,
)
from .seed import (
    item_rng,
    item_seed,
    preserve_determinism_flags,
    seed_everything,
    stable_seed,
    torch_available,
)
from .splits import (
    SelectionDeclaration,
    assert_classes_match,
    assert_split_disjoint,
    assert_training_pool_equality,
    declare_selection_set,
    present_classes,
)
from .triage import BELOW_BAR, CLOSE, HOLD, triage

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # seed
    "seed_everything", "stable_seed", "item_seed", "item_rng",
    "preserve_determinism_flags", "torch_available",
    # controls
    "ControlResult", "ControlSuite", "label_permutation_control", "overfit_one_batch",
    "overfit_one_batch_generic", "select_signal_batch", "BEST_CHECKPOINT_KEY",
    # splits
    "assert_training_pool_equality", "assert_split_disjoint", "present_classes",
    "assert_classes_match", "declare_selection_set", "SelectionDeclaration",
    # triage
    "triage", "CLOSE", "HOLD", "BELOW_BAR",
    # baselines
    "trivial_baselines", "measured_chance", "stated_floor",
    # convergence
    "convergence_evidence", "assert_sweep_converged",
    # artifacts
    "run_key", "run_dir", "guard_overwrite", "write_json_guarded",
    # dedup
    "md5_file", "md5_groups", "sign_hash_from_images", "hash_pairs_below",
    "connected_components", "reencode_positive_control", "assert_positive_control_ran",
]
