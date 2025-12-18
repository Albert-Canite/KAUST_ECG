# ECG Four-Class Training Pipeline

This project trains a lightweight segment-aware 1D CNN to classify ECG beats into four AAMI categories (N, S, V, O) from the MIT-BIH Arrhythmia Database.

## Requirements
- Python 3.8
- `wfdb`, `numpy`, `scikit-learn`, `torch` (tested with PyTorch >= 1.10), `tqdm`, `matplotlib`, `seaborn`

## Directory
- `main.py`: entrypoint for training and evaluation
- `data.py`: beat extraction and dataset utilities
- `model.py`: segment-aware 1D CNN
- `metrics.py`: metrics, confusion matrix, logging helpers
- `sampler.py`: balanced sampling, effective-number class weights, sanity check
- `utils.py`: reproducibility, JSON IO, argument helpers

## Quickstart
```bash
python main.py
```
`main.py` defaults to the MIT-BIH root `E:/OneDrive - KAUST/ONN codes/MIT-BIH/mit-bih-arrhythmia-database-1.0.0/`. Override with `--data_root` only if your path differs.

### Common flags
- `--val_ratio 0.2` controls the validation split by record (fraction of TRAIN_RECORDS used for validation).
- `--seed 42` sets the random seed for reproducibility.
- `--normalization zscore` or `robust` controls beat normalization.
- `--tau 1.0` sets the logit adjustment temperature; `--beta 0.9999` sets the effective number beta.

The script automatically evaluates on the held-out generalization records (`GENERALIZATION_RECORDS`) every epoch. Artifacts such as `best.pt`, `last.pt`, `results.json`, `train.log`, metric curves (`metrics_tracking.png`), and the best validation/generalization confusion matrices are saved to the `outputs/` directory.
