# MIT-BIH Segment-Aware ECG Classification & Distillation

Single-lead beat classification on the MIT-BIH Arrhythmia Database with a compact segment-aware student and an optional 1D ResNet teacher for knowledge distillation. The codebase provides:

- `train.py`: baseline student training with class rebalancing, recall-biased threshold sweeps, and validation/generalization reporting.
- `KD.py`: end-to-end distillation entrypoint that reuses the same dataloaders/model builder, auto-trains or loads a teacher, gates KD by teacher quality, logs training, and plots ROC comparisons for both validation and generalization splits.
- `models/`: segment-aware student definition and constrained layers for bounded weights/activations.

## Data & Preprocessing
- Dataset: MIT-BIH Arrhythmia Database. Set `--data_path` to the folder containing `*.dat`/`*.hea` files.
- Beats: 360-sample windows centered on annotations; per-beat mean removal and max-abs scaling to `[-1, 1]` (see `data.py`).
- Splits: `TRAIN_RECORDS` vs. `GENERALIZATION_RECORDS` (hold-out) with a 80/20 train/val split inside `TRAIN_RECORDS`.

## Models & Parameter Counts
### Student (models/student.py)
- Inputs: `(batch, 1, 360)` sliced into P/QRS/T/global segments.
- Encoders: four `Conv1d(1, 4, kernel_size=4, stride=1)` blocks (value-constrained by default) → pooled into 8 tokens of dim 4.
- MLP head: `num_mlp_layers` × `Linear(4, 4)` with ReLU/tanh + token dropout, followed by mean pooling and dropout.
- Classifier: `Linear(4, 2)`.
- Minimum depth: **now allows a single MLP layer** (`num_mlp_layers=1` keeps one block; previously clamped to 2).
- Parameter counts (includes biases):
  - `num_mlp_layers=1`: **110** parameters.
  - `num_mlp_layers=2`: **130** parameters.
  - `num_mlp_layers=3` (default): **150** parameters.

### Teacher (KD.py)
- Architecture: 1D ResNet18 variant with a (Conv-BN-ReLU-MaxPool) stem, four residual stages [32, 64, 128, 256] with two BasicBlock1D units each, global average pooling, and `Linear(256, 2)` classifier.
- Parameters: **964,002**.
- Outputs logits and a pooled embedding for feature-level KD.

## Baseline Training (`train.py`)
- Class imbalance handling: per-epoch class weights from `compute_class_weights`; optional weighted/balanced samplers triggered when abnormal ratio is low.
- Objective & optimization: cross-entropy, `Adam` with `ReduceLROnPlateau` on validation loss; gradient clipping to 1.0.
- Threshold tuning: `sweep_thresholds_blended` sweeps candidate thresholds to satisfy `miss <= threshold_target_miss` and `fpr <= threshold_max_fpr`, blending validation/generalization metrics via `generalization_score_weight` and recall/miss penalties. The best threshold is stored in the checkpoint and reused for reporting.
- Metrics: loss, F1, miss rate, FPR on validation and generalization splits each epoch; final probabilities are persisted for offline analysis.
- Checkpoints: saved to `saved_models/student_model.pth` with configuration and best threshold.

### Example
```bash
python train.py --data_path /path/to/mit-bih --batch_size 128 --num_mlp_layers 1 --dropout_rate 0.2 \
  --threshold_target_miss 0.10 --threshold_max_fpr 0.10
```

## Knowledge Distillation (`KD.py`)
- Baseline evaluation: loads `student_model.pth`, rebuilds the student with its saved config, and reports baseline metrics on val/gen with their tuned threshold.
- Teacher: builds `ResNet1D18`; trains with class-weighted cross-entropy if `teacher_model.pth` is absent, otherwise loads it. Teacher thresholds are swept on both val/gen to pick `teacher_thr`.
- KD gating: compares teacher generalization miss/FPR at `teacher_thr` against the baseline student. If teacher miss is higher or FPR exceeds `max(student_gen_fpr, threshold_max_fpr)`, KD losses are disabled; otherwise KD proceeds. Effective KD weights (`kd_alpha`, `kd_beta`) and `use_kd` are printed.
- Losses: cross-entropy with positive upweighting plus optional logit KL (temperature-scaled) and feature MSE distillation. KD can be toggled off dynamically (`use_kd=False`) without changing CE training.
- Logging & artifacts: per-epoch KD metrics written to `artifacts/kd_training_log.csv`; ROC curves for validation and generalization (baseline student vs. teacher vs. KD student) saved to `figures/kd_roc_comparison.png`.
- Outputs: distilled checkpoint `saved_models/student_KD.pth` with its own best threshold from the KD run.

### Example
```bash
python KD.py --data_path /path/to/mit-bih --student_path saved_models/student_model.pth \
  --teacher_path saved_models/teacher_model.pth --kd_gen_mix_ratio 0.35 --num_mlp_layers 1
```

## Notes on Capacity Choices
- The student is extremely compact; dropping to a single MLP layer (110 params) further reduces capacity for deployment. Higher `num_mlp_layers` modestly increases parameters while keeping the footprint tiny (<200 params).
- The teacher remains ~0.96M parameters to provide a stronger signal when its generalization miss/FPR beat the student; KD is automatically skipped otherwise to avoid harming recall.

## Artifacts & Logging
- Baseline training: artifacts in `./artifacts` (probabilities, plots) and checkpoints in `./saved_models`.
- Distillation: `artifacts/kd_training_log.csv` plus ROC comparison figure in `./figures`.

