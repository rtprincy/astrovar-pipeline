
# astrovar-pipeline

A clean, modular ML pipeline that integrates your existing scripts to:
1) Extract Gaia DR3 light curves from a list of source IDs
2) Run a hybrid frequency search (LS + LK / Ψ-statistic) on each light curve
3) Compute light-curve & periodogram features, plus augment with Gaia summary statistics
4) De-correlate & select features; embed with t-SNE; cluster with GMM; prune with RF importance; re-embed/re-cluster; visualize.

## Quick start
```bash
>> git clone https://github.com/rtprincy/astrovar-pipeline.git
>> cd astrovar-pipeline

>> python3 -m venv .venv
>> source .venv/bin/activate       # (on macOS/Linux)
# or
>> .venv\Scripts\activate          # (on Windows PowerShell)
>> pip install -e .

>> export GAIA_USER="your_gaia_username"
>> export GAIA_PASS="your_gaia_password"

>> astrovar run --config astrovar_pipeline/configs/defaults.yaml
```

Outputs land under `./outputs/` by default.
Edit `astrovar_pipeline/configs/defaults.yaml` to set Gaia credentials, data paths, and hyperparameters.

You can also run individual stages, e.g.:

### To extract Gaia light curves:
```bash
>> astrovar extract --config astrovar_pipeline/configs/defaults.yaml 
```
### To run frequency search:
```bash
>> astrovar freq --config astrovar_pipeline/configs/defaults.yaml
```
### To extract features:
```bash
>> astrovar features --config astrovar_pipeline/configs/defaults.yaml
```
### To cluster objects:
```bash
>> astrovar cluster --config astrovar_pipeline/configs/defaults.yaml
```

## References:
For more details on the unsupervised clustering methods using t-SNE, please see the following papers:

Ranaivomanana et al. (2025a): https://arxiv.org/abs/2411.18609
Ranaivomanana et al. (2025b): https://arxiv.org/abs/2510.23776

