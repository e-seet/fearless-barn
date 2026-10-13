# RETROSPECTIVE — Arm: mlp_antigravity

## 1. Motivation
`mlp_antigravity` implements a PyTorch deep tabular neural network with entity embeddings for categorical features (`stand_runway`, `ADEP_mvt`, `AIRCRAFT_TYPE_mvt`, etc.) combined with batch-normalized continuous features.

Unlike decision trees (LightGBM & XGBoost), neural networks partition feature space smoothly and construct low-dimensional metric embeddings for categorical combinations. This provides structural diversity for ensembling.

## 2. Architecture & Contracts
- Entity Embeddings: Dimension $d = \min(48, \max(4, \text{int}(C^{0.5})))$
- Backbone: Multi-layer feed-forward network with BatchNorm, GELU, and Dropout
- Loss: MSE Loss matching the L2 RMSE metric
- Accelerator: NVIDIA CUDA with AdamW optimizer and gradient clipping
