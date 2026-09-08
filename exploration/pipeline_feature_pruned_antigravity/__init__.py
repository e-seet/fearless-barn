"""pipeline_feature_pruned_antigravity: Feature Ablation and Pruning arm.

Evaluates performance gains from eliminating noisy out-of-range temporal features
(month_num, doy, doy_sin, doy_cos) and high-variance sparse interactions.
"""
from __future__ import annotations

__all__ = []
