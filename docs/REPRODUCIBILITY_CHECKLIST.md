# Reproducibility Checklist

- [ ] Random seeds fixed and documented per module (`SEED=42` unless noted)
- [ ] Package versions pinned in `requirements.txt` / `environment.yml`
- [ ] Train/test split boundaries documented in `layer2_anomaly_detection/run.py`
- [ ] Expected runtime documented per stage (see README "Reproducing every number" section)
- [ ] All figures regenerate byte-identical `results/figures/*.png` from committed `results/*.csv|json|parquet`
- [ ] CI workflow (`.github/workflows/reproduce.yml`) runs the full pipeline on every push to `main`
