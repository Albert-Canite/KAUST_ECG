# Segment-Aware MIT-BIH ECG Classification

This project provides a fully configurable training pipeline for single-lead beat classification on the MIT-BIH Arrhythmia Database using a lightweight segment-aware student model and an optional ResNet18-based teacher for knowledge distillation. Value-constrained layers and activation scaling are available to bound weights/activations for deployment in limited numeric domains.

## Repository Layout
- `train.py` – Training entrypoint with argument-parsable hyperparameters, early stopping, LR scheduling, class weighting, class-rebalancing sampler, and optional distillation with teacher quality checks.
- `data.py` – MIT-BIH loading, per-beat preprocessing (mean removal + max-abs scaling to `[-1, 1]`), and dataset split utilities.
- `models/student.py` – Segment-aware student encoder that slices beats into P/QRS/T/Global windows, produces eight 4-D tokens, feeds a configurable photonic MLP, and classifies beats.
- `models/teacher.py` – 1D ResNet18 teacher producing logits and an intermediate embedding for distillation.
- `constraints.py` – Tanh-reparameterized Conv1d/Linear layers plus activation scaling helpers for bounded weights/inputs.
- `utils.py` – Class-weight computation, confusion metrics, and KD logit loss helpers.

## Installation
1. Install Python 3.9+ and PyTorch (with CUDA if available).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # if present
   pip install wfdb numpy torch seaborn matplotlib
   ```

## Data
Download the MIT-BIH Arrhythmia Database and set `--data_path` to the folder containing record files (e.g., `100.dat`, `100.hea`). Each beat is extracted as a 360-sample window centered on annotations and normalized per beat.

## Training
### One-click / default run
Simply run the script (e.g., click "Run" in an IDE or execute `python train.py`). By default, the student trains **with knowledge distillation enabled**, but KD stays off for the first few epochs (`--kd_warmup_epochs`, default 5) until the student stabilizes. If no teacher checkpoint is provided, a compact ResNet18-1D teacher is auto-trained (15 epochs by default), validated, and only used for KD if it meets minimal F1/TPR thresholds; otherwise KD is disabled to avoid harming recall. Checkpoints and the auto-trained teacher are saved under `saved_models/`.

### Command-line customization
- Basic student-only training:
  ```bash
  python train.py --data_path /path/to/mit-bih --max_epochs 90 --no-use-kd
  ```

- Knowledge distillation with an existing teacher:
  ```bash
  python train.py --data_path /path/to/mit-bih --use-kd \
    --teacher_checkpoint /path/to/teacher.pth --teacher_embedding_dim 128 \
    --kd_temperature 2.0 --kd_d 16 --alpha 1.0 --beta 1.0 --gamma 1.0
  ```

- Enable bounded-weight/value pipeline:
  ```bash
  python train.py --data_path /path/to/mit-bih --use_value_constraint --use_tanh_activations \
    --constraint_scale 1.0 --dropout_rate 0.1
  ```

Early stopping monitors a recall-biased score (`F1 + 1.5*sensitivity - FPR`) with configurable patience (`--patience`, default 25) and a minimum epoch guard (`--min_epochs`, default 25). Learning-rate scheduling uses `ReduceLROnPlateau` on the validation loss (`--scheduler_patience`, default 3, `factor=0.5`). Gradients are clipped to `max_norm=1.0`.

**Threshold tuning**: each validation pass sweeps a dense grid plus probability quantiles in `[0.05, 0.95]`, first filtering thresholds that satisfy `miss <= --threshold_target_miss` (default 0.12) and `fpr <= --threshold_max_fpr` (default 0.20). Among feasible candidates it maximizes `F1 + 1.5*sensitivity - FPR`; if none meet the constraints, the best score over all thresholds is used. The chosen threshold drives early stopping, artifact reporting, generalization evaluation, and is stored in the checkpoint.

**Class imbalance handling**: training begins with **no class weights and no weighted sampler** for a short warmup (`--imbalance_warmup_epochs`, default 5) to avoid early collapse, then switches to mild reweighting (abnormal weight `--class_weight_abnormal=1.2`, ratio clamp `--max_class_weight_ratio=2.0`). Weighted sampling is auto-enabled after warmup when the abnormal ratio is below `--auto_sampler_ratio` (default 0.35) or when `--use_weighted_sampler` is set, using boost `--sampler_abnormal_boost` (default 1.2). Adaptive recall rescue triggers when miss exceeds `--recall_target_miss` (default 0.15) under an FPR cap (`--adaptive_fpr_cap`, default 0.25) and can fire up to `--recall_rescue_limit` times (default 3), increasing abnormal emphasis and enabling the sampler. Dual collapse detectors pause KD and drop rebalancing if the model predicts nearly all abnormal (`FPR>95% & miss<5%`) or nearly all normal (`miss>95% & FPR<5%`).

## Segment-Aware Student Overview
- Inputs: `(batch_size, 1, 360)`
- Four Conv1d encoders (P/QRS/T/Global): `Conv1d(1, 4, kernel_size=4, stride=1, padding=0)`
- Token pooling: P→2 tokens, QRS→3 tokens, T→2 tokens (AvgPool over time), Global→1 token (global average); concatenated tokens shape `(batch, 8, 4)`
- Photonic MLP: `num_mlp_layers` (>=2) of `Linear(4, 4) + ReLU` (optionally tanh with constrained weights)
- Token average pooling → `h_pool` `(batch, 4)` → optional dropout → classifier `Linear(4, 2)`

## Knowledge Distillation
- Teacher: 1D ResNet18 producing logits and an embedding (`embedding_dim`).
- Student: logits + pooled feature (`h_pool`).
- Loss: `alpha * CE` + `beta * KL(softmax(z_T/T) || softmax(z_S/T))` + `gamma * MSE(norm(proj_T(feat_T)), norm(proj_S(h_pool)))`.
- Projections: `proj_T: Linear(embedding_dim, kd_d)`, `proj_S: Linear(4, kd_d)` (constrained if `--use_value_constraint`).
- Teacher quality guard: KD is automatically disabled if the teacher validation F1 or sensitivity drops below `--teacher_min_f1/--teacher_min_sensitivity`; KD also pauses when validation miss exceeds `--kd_pause_miss` and resumes after it falls below `--kd_resume_miss`.

## Value Constraints & Normalization
- Constrained layers use tanh-reparameterized weights (and optional bias) scaled by `constraint_scale` to keep weights in `[-scale, scale]`.
- Inputs to constrained layers can be scaled to `[-1, 1]` via `scale_to_unit`; alternatively, tanh activations bound them directly (`--use_tanh_activations`).
- Beat preprocessing keeps raw inputs in `[-1, 1]` via mean removal and max-abs scaling; optional post-layer dropout mitigates overfitting.

## Saved Artifacts
Checkpoints are written to `saved_models/student_model.pth` with the CLI configuration. Training curves, ROC curves (val/generalization), and confusion matrices are exported to `./artifacts` as PNG files. Adjust paths as needed for your environment.
