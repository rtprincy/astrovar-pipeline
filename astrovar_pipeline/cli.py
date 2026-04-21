from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import pandas as pd
from .utils.config import load_config
from .io.gaia import login, read_source_ids, fetch_gaia_epoch_photometry
from .periodogram.psi_search import psi_periodogram, freq_grid
from .features.extract import (
    extract_features_from_lc,
    PeriodogramBundle,
    augment_with_gaia_summary,
)
from .models.pipeline import (
    remove_correlated,
    optimize_tsne,
    optimize_gmm,
    rf_prune_features,
)
from .viz.plots import plot_tsne, plot_importances
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm


def app():
    parser = argparse.ArgumentParser(prog="astrovar")
    parser.add_argument(
        "command",
        choices=["run", "extract", "freq", "features", "cluster"],
        help="Which stage to run",
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "configs" / "defaults.yaml")
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sids = read_source_ids(cfg["data"]["source_id_list"])

    outdir = Path(cfg["data"]["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    cache = Path(cfg["data"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)

    if args.command in ("run", "extract"):
        login(cfg["gaia"].get("username"), cfg["gaia"].get("password"))
        fetch_gaia_epoch_photometry(sids, outdir / "lightcurves")
        if args.command != "run":
            return

    if args.command in ("run", "freq"):
        lc_dir = outdir / "lightcurves"
        per_dir = outdir / "periodograms"
        per_dir.mkdir(exist_ok=True)

        rows = []
        lcs = [p for p in os.listdir(lc_dir) if int(p.strip(".csv")) in sids]
        for path in tqdm(lcs, "Processing lightcurves"):
            best_frequencies = []
            sid = int(path.strip(".csv"))

            if not os.path.exists(per_dir / f"{sid}_rp.npz"):
                lc = pd.read_csv(lc_dir / path)

                # check if minimum lightcurve length is met
                if len(lc.dropna()) < cfg["extraction"]["min_points"]:
                    print("Not enough observations in lightcurve.")
                    continue

                # calculate and save the periodogram for every filter
                for filter in ["g", "bp", "rp"]:
                    lc_x = lc[~lc[f"variability_flag_{filter}_reject"].values]

                    ext = "_transit" if filter == "g" else ""
                    time_ext = "_obs" if filter != "g" else ""

                    mag = lc_x[f"{filter + ext}_mag"].dropna().to_numpy(copy=True)
                    time = (
                        lc_x[f"{filter + ext + time_ext}_time"]
                        .dropna()
                        .to_numpy(copy=True)
                    )
                    flux = lc_x[f"{filter + ext}_flux"].dropna().to_numpy(copy=True)
                    flux_err = (
                        lc_x[f"{filter + ext}_flux_error"].dropna().to_numpy(copy=True)
                    )
                    mag_err = (2.5 / np.log(10)) * (flux_err / flux)

                    per_x = psi_periodogram(
                        time,
                        mag,
                        mag_err,
                        cfg["frequency_search"]["min_freq"],
                        cfg["frequency_search"]["max_freq"],
                        cfg["frequency_search"]["oversample"],
                    )

                    best_frequencies.append(
                        float(per_x["freq"][np.nanargmax(per_x["psi"])])
                    )

                    np.savez(per_dir / f"{sid}_{filter}.npz", **per_x)
            else:
                for filter in ["g", "bp", "rp"]:
                    per_x = np.load(per_dir / f"{sid}_{filter}.npz")
                    best_frequencies.append(per_x["freq"][np.nanargmax(per_x["psi"])])

            rows.append(
                {
                    "source_id": sid,
                    "best_freq_g": best_frequencies[0],
                    "best_freq_bp": best_frequencies[1],
                    "best_freq_rp": best_frequencies[2],
                }
            )

            pd.DataFrame(rows).to_csv(outdir / "periodogram_summary.csv", index=False)
        if args.command != "run":
            return

    if args.command in ("run", "features"):
        lc_dir = outdir / "lightcurves"
        per_dir = outdir / "periodograms"
        feat_rows = []
        for path in lc_dir.glob("*.csv"):
            sid = int(path.stem)
            lc = pd.read_csv(path)
            npz_g = per_dir / f"{sid}_g.npz"
            npz_bp = per_dir / f"{sid}_bp.npz"
            npz_rp = per_dir / f"{sid}_rp.npz"

            per_g = None
            per_bp = None
            per_rp = None

            if npz_g.exists():
                z_g = np.load(npz_g)
                per_g = PeriodogramBundle(
                    freq=z_g["freq"], lsp=z_g["lsp"], theta=z_g["theta"]
                )
            if npz_bp.exists():
                z_bp = np.load(npz_bp)
                per_bp = PeriodogramBundle(
                    freq=z_bp["freq"], lsp=z_bp["lsp"], theta=z_bp["theta"]
                )
            if npz_bp.exists():
                z_rp = np.load(npz_rp)
                per_rp = PeriodogramBundle(
                    freq=z_rp["freq"], lsp=z_rp["lsp"], theta=z_rp["theta"]
                )

            feats = extract_features_from_lc(
                cfg["frequency_search"]["oversample"], lc, per_g, per_bp, per_rp
            )
            feats["source_id"] = sid
            feat_rows.append(feats)

        feats_df = pd.DataFrame(feat_rows).set_index("source_id")
        # optional: augment with Gaia summary stats
        if cfg["features"]["augment_with_gaia_summary"]:
            extra = augment_with_gaia_summary(list(feats_df.index))
            feats_df = feats_df.join(extra, how="left")

        feats_df.to_csv(outdir / "features_raw.csv")
        if args.command != "run":
            return

    if args.command in ("run", "cluster"):
        feats_df = pd.read_csv(outdir / "features_raw.csv")
        feats_df = feats_df.drop(columns=["source_id"])
        print("Feature list:", feats_df.columns.values)
        # X = feats_df.select_dtypes("number").fillna(feats_df.median(numeric_only=True))
        X = feats_df.select_dtypes("number").dropna()
        X1, dropped = remove_correlated(X, cfg["ml"]["correlation_threshold"])
        scaler = StandardScaler()
        X1_norm = scaler.fit_transform(X1)
        print("Number of reduced features: ", X1.shape[1])
        emb, tsne_params = optimize_tsne(X1_norm, cfg["ml"]["tsne"])
        X1_norm = pd.DataFrame(X1_norm, columns=X1.columns)
        labels, gmm_cfg, gmm_model = optimize_gmm(emb, cfg["ml"]["gmm"])
        # RF pruning
        if cfg["ml"]["rf_prune"]["enabled"]:
            X2, importances = rf_prune_features(
                X1_norm,
                labels,
                top_k=cfg["ml"]["rf_prune"]["top_k_features"],
                random_state=cfg["ml"]["rf_prune"]["random_state"],
            )
            emb2, tsne_params2 = optimize_tsne(X2.to_numpy(), cfg["ml"]["tsne"])
            labels2, gmm_cfg2, gmm_model2 = optimize_gmm(emb2, cfg["ml"]["gmm"])
            importances.to_csv(outdir / "rf_importances.csv")
            plot_importances(importances, str(outdir / "rf_importances_top30.png"))
            np.save(outdir / "tsne_embedding.npy", emb2)
            pd.DataFrame({"source_id": X1_norm.index, "label": labels2}).to_csv(
                outdir / "cluster_labels.csv", index=False
            )
            plot_tsne(emb2, labels2, str(outdir / "tsne_clusters.png"))
        else:
            np.save(outdir / "tsne_embedding.npy", emb)
            pd.DataFrame({"source_id": X1_norm.index, "label": labels}).to_csv(
                outdir / "cluster_labels.csv", index=False
            )
            plot_tsne(emb, labels, str(outdir / "tsne_clusters.png"))
