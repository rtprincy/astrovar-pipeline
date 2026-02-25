from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import pandas as pd
from .utils.config import load_config
from .io.gaia import login, read_source_ids, fetch_gaia_epoch_photometry
from .periodogram.psi_search import psi_periodogram, freq_grid
from astropy.timeseries import LombScargle
from lk_stat_package import lk_stat
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
    outdir = Path(cfg["data"]["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["data"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)

    if args.command in ("run", "extract"):
        login(cfg["gaia"].get("username"), cfg["gaia"].get("password"))
        sids = read_source_ids(cfg["data"]["source_id_list"])
        fetch_gaia_epoch_photometry(sids, outdir / "lightcurves")
        if args.command != "run":
            return

    if args.command in ("run", "freq"):
        lc_dir = outdir / "lightcurves"
        per_dir = outdir / "periodograms"
        per_dir.mkdir(exist_ok=True)
        rows = []
        lcs = list(lc_dir.glob("*csv"))
        for p in tqdm(lcs, "Processing lightcurves"):
            sid = int(p.stem)

            if not os.path.exists(per_dir / f"{sid}_rp.npz"):
                lc = pd.read_csv(p)

                # lc_flag=(lc['variability_flag_g_reject'].values)&(lc['variability_flag_bp_reject'].values)&(lc['variability_flag_rp_reject'].values)
                lc_g = lc[~lc["variability_flag_g_reject"].values]
                lc_bp = lc[~lc["variability_flag_bp_reject"].values]
                lc_rp = lc[~lc["variability_flag_rp_reject"].values]

                # Compute periodogram on g-band subset if available
                lc = lc.dropna()
                if lc.shape[0] < cfg["extraction"]["min_points"]:
                    continue

                mag_g = lc_g["g_transit_mag"].dropna().to_numpy(copy=True)
                Time_g = lc_g["g_transit_time"].dropna().to_numpy(copy=True)
                flux_g = lc_g["g_transit_flux"].dropna().to_numpy(copy=True)
                flux_err_g = lc_g["g_transit_flux_error"].dropna().to_numpy(copy=True)
                mag_err_g = (2.5 / np.log(10)) * (flux_err_g / flux_g)

                mag_bp = lc_bp["bp_mag"].dropna().to_numpy(copy=True)
                Time_bp = lc_bp["bp_obs_time"].dropna().to_numpy(copy=True)
                flux_bp = lc_bp["bp_flux"].dropna().to_numpy(copy=True)
                flux_err_bp = lc_bp["bp_flux_error"].dropna().to_numpy(copy=True)
                mag_err_bp = (2.5 / np.log(10)) * (flux_err_bp / flux_bp)

                mag_rp = lc_rp["rp_mag"].dropna().to_numpy(copy=True)
                Time_rp = lc_rp["rp_obs_time"].dropna().to_numpy(copy=True)
                flux_rp = lc_rp["rp_flux"].dropna().to_numpy(copy=True)
                flux_err_rp = lc_rp["rp_flux_error"].dropna().to_numpy(copy=True)
                mag_err_rp = (2.5 / np.log(10)) * (flux_err_rp / flux_rp)

                # print("G-band")
                per_g = psi_periodogram(
                    Time_g,
                    mag_g,
                    mag_err_g,
                    cfg["frequency_search"]["min_freq"],
                    cfg["frequency_search"]["max_freq"],
                    cfg["frequency_search"]["oversample"],
                )

                # print("BP-band")
                per_bp = psi_periodogram(
                    Time_bp,
                    mag_bp,
                    mag_err_bp,
                    cfg["frequency_search"]["min_freq"],
                    cfg["frequency_search"]["max_freq"],
                    cfg["frequency_search"]["oversample"],
                )

                # print("RP-band")
                per_rp = psi_periodogram(
                    Time_rp,
                    mag_rp,
                    mag_err_rp,
                    cfg["frequency_search"]["min_freq"],
                    cfg["frequency_search"]["max_freq"],
                    cfg["frequency_search"]["oversample"],
                )

                # per_g = psi_periodogram(lc["g_transit_time"].values, lc["g_transit_flux"].values, lc["g_transit_flux_error"].values,
                #                   cfg["frequency_search"]["min_freq"], cfg["frequency_search"]["max_freq"], cfg["frequency_search"]["oversample"])

                # per_bp = psi_periodogram(lc["bp_obs_time"].values, lc["bp_flux"].values, lc["bp_flux_error"].values,
                #                   cfg["frequency_search"]["min_freq"], cfg["frequency_search"]["max_freq"], cfg["frequency_search"]["oversample"])

                # per_rp = psi_periodogram(lc["rp_obs_time"].values, lc["rp_flux"].values, lc["rp_flux_error"].values,
                #                   cfg["frequency_search"]["min_freq"], cfg["frequency_search"]["max_freq"], cfg["frequency_search"]["oversample"])
                np.savez(per_dir / f"{sid}_g.npz", **per_g)
                np.savez(per_dir / f"{sid}_bp.npz", **per_bp)
                np.savez(per_dir / f"{sid}_rp.npz", **per_rp)

            else:

                per_g = np.load(per_dir / f"{sid}_g.npz")
                per_bp = np.load(per_dir / f"{sid}_bp.npz")
                per_rp = np.load(per_dir / f"{sid}_rp.npz")

            rows.append(
                {
                    "source_id": sid,
                    "best_freq_g": float(per_g["freq"][np.nanargmax(per_g["psi"])]),
                    "best_freq_bp": float(per_bp["freq"][np.nanargmax(per_bp["psi"])]),
                    "best_freq_rp": float(per_rp["freq"][np.nanargmax(per_rp["psi"])]),
                }
            )

            pd.DataFrame(rows).to_csv(outdir / "periodogram_summary.csv", index=False)
        if args.command != "run":
            return

    if args.command in ("run", "features"):
        lc_dir = outdir / "lightcurves"
        per_dir = outdir / "periodograms"
        feat_rows = []
        for p in lc_dir.glob("*.csv"):
            sid = int(p.stem)
            lc = pd.read_csv(p)
            npz_g = per_dir / f"{sid}_g.npz"
            npz_bp = per_dir / f"{sid}_bp.npz"
            npz_rp = per_dir / f"{sid}_rp.npz"

            per_g = None
            per_bp = None
            per_rp = None

            if npz_g.exists():
                z_g = np.load(npz_g)
                per_g = PeriodogramBundle(
                    freq=z_g["freq"], lsp=z_g["lsp"], theta=z_g["theta"], psi=z_g["psi"]
                )
            if npz_bp.exists():
                z_bp = np.load(npz_bp)
                per_bp = PeriodogramBundle(
                    freq=z_bp["freq"],
                    lsp=z_bp["lsp"],
                    theta=z_bp["theta"],
                    psi=z_bp["psi"],
                )
            if npz_bp.exists():
                z_rp = np.load(npz_rp)
                per_rp = PeriodogramBundle(
                    freq=z_rp["freq"],
                    lsp=z_rp["lsp"],
                    theta=z_rp["theta"],
                    psi=z_rp["psi"],
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
