
# astrovar-pipeline

A clean, modular ML pipeline that integrates your existing scripts to:
1) Extract Gaia DR3 light curves from a list of source IDs
2) Run a hybrid frequency search (LS + LK / Ψ-statistic) on each light curve
3) Compute light-curve & periodogram features, plus augment with Gaia summary statistics
4) De-correlate & select features; embed with t-SNE; cluster with GMM; prune with RF importance; re-embed/re-cluster; visualize.

## Quick start
```bash
pip install -e .
astrovar run --config configs/defaults.yaml
```

Outputs land under `./outputs/` by default.
Edit `configs/defaults.yaml` to set Gaia credentials, data paths, and hyperparameters.
