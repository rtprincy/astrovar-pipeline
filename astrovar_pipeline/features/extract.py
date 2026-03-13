from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

# from .lc_stats import LCStats
from .extract_gaia_parameters import query_gaia_params, query_vari_summary
from .LCStatistics import LCStatistics as lcs
from astroquery.gaia import Gaia
from astropy.timeseries import LombScargle
from lk_stat_package import lk_stat


@dataclass
class PeriodogramBundle:
    freq: np.ndarray
    lsp: np.ndarray
    theta: np.ndarray


def freq_grid(times, oversample_factor=10, f0=None, fn=None):
    times = np.sort(times)
    df = 1.0 / (times.max() - times.min())
    if (f0 is None) | (int(f0) == 0):
        f0 = df
    if fn is None:
        fn = 0.5 / np.median(np.diff(times))
    return np.arange(f0, fn, df / oversample_factor)


def optimise_freq(oversampling_factor, psi, freq, Time, mag, mag_err):
    idx_peak = np.argmax(psi)
    f_step = np.diff(freq)[0]
    peak_freq = freq[idx_peak]
    peak_period = 1 / peak_freq
    # Here we take 10 (oversampling factor) steps before and after the frequency peak as a new frequency search range.
    lower_range = max(freq.min(), peak_freq - (oversampling_factor * f_step))
    upper_range = min(freq.max(), peak_freq + (oversampling_factor * f_step))
    fine_grid_freq = freq_grid(
        Time, oversample_factor=100, f0=lower_range, fn=upper_range
    )
    lsp_fg = LombScargle(t=Time, y=mag, dy=mag_err, nterms=1).power(
        frequency=fine_grid_freq, method="cython", normalization="psd"
    )
    theta_fg = lk_stat(1 / fine_grid_freq, mag, mag_err, Time)
    psi_fine_grid = 2 * lsp_fg / theta_fg
    best_freq = fine_grid_freq[np.argmax(psi_fine_grid)]
    best_period = 1 / best_freq

    return best_freq, best_period


def extract_features_from_lc(
    oversampling_factor: np.int32,
    lc: pd.DataFrame,
    per_g: PeriodogramBundle | None = None,
    per_bp: PeriodogramBundle | None = None,
    per_rp: PeriodogramBundle | None = None,
) -> Dict[str, Any]:
    feats = {}
    # Add periodogram features
    if (per_g is not None) & (per_bp is not None) & (per_rp is not None):
        lc_g = lc[~lc["variability_flag_g_reject"].values]
        lc_bp = lc[~lc["variability_flag_bp_reject"].values]
        lc_rp = lc[~lc["variability_flag_rp_reject"].values]

        mag_g = lc_g["g_transit_mag"].dropna().values
        Time_g = lc_g["g_transit_time"].dropna().values
        flux_g = lc_g["g_transit_flux"].dropna().values
        flux_err_g = lc_g["g_transit_flux_error"].dropna().values
        mag_err_g = (2.5 / np.log(10)) * (flux_err_g / flux_g)

        mag_bp = lc_bp["bp_mag"].dropna().values
        Time_bp = lc_bp["bp_obs_time"].dropna().values
        flux_bp = lc_bp["bp_flux"].dropna().values
        flux_err_bp = lc_bp["bp_flux_error"].dropna().values
        mag_err_bp = (2.5 / np.log(10)) * (flux_err_bp / flux_bp)

        mag_rp = lc_rp["rp_mag"].dropna().values
        Time_rp = lc_rp["rp_obs_time"].dropna().values
        flux_rp = lc_rp["rp_flux"].dropna().values
        flux_err_rp = lc_rp["rp_flux_error"].dropna().values
        mag_err_rp = (2.5 / np.log(10)) * (flux_err_rp / flux_rp)

        best_freq_g, best_period_g = optimise_freq(
            oversampling_factor, per_g.psi, per_g.freq, Time_g, mag_g, mag_err_g
        )
        best_freq_bp, best_period_bp = optimise_freq(
            oversampling_factor, per_bp.psi, per_bp.freq, Time_bp, mag_bp, mag_err_bp
        )
        best_freq_rp, best_period_rp = optimise_freq(
            oversampling_factor, per_rp.psi, per_rp.freq, Time_rp, mag_rp, mag_err_rp
        )

        # best_freq_g=per_g.freq[np.nanargmax(per_g.psi)]
        # best_freq_bp=per_bp.freq[np.nanargmax(per_bp.psi)]
        # best_freq_rp=per_rp.freq[np.nanargmax(per_rp.psi)]

        # best_period_g=1/best_freq_g
        # best_period_bp=1/best_freq_bp
        # best_period_rp=1/best_freq_rp

        std_period = np.std([best_period_g, best_period_bp, best_period_rp])
        frac_period = (best_period_g) / std_period

        featG = lcs(
            "G",
            mag_g,
            mag_err_g,
            Time_g,
            per_g.lsp,
            per_g.psi,
            per_g.freq,
            flux_g,
            flux_err_g,
            best_freq_g,
        )
        featBP = lcs(
            "BP",
            mag_bp,
            mag_err_bp,
            Time_bp,
            per_bp.lsp,
            per_bp.psi,
            per_bp.freq,
            flux_bp,
            flux_err_bp,
            best_freq_bp,
        )
        featRP = lcs(
            "RP",
            mag_rp,
            mag_err_rp,
            Time_rp,
            per_rp.lsp,
            per_rp.psi,
            per_rp.freq,
            flux_rp,
            flux_err_rp,
            best_freq_rp,
        )

        feats.update(featG.compute_all_parameters())
        feats.update(featBP.compute_all_parameters())
        feats.update(featRP.compute_all_parameters())

        feats.update(
            {
                "std": std_period,
                "frac_period": frac_period,
                "opt_period_g": best_period_g,
                "opt_period_bp": best_period_bp,
                "opt_period_rp": best_period_rp,
            }
        )

        # print("Extracting Gaia summary variability statistics")

        # Manually compute Gaia variability statistics

        # gaia_summary_stat_G=calculate_variability_metrics(Time_g, mag_g, mag_err_g, band_suffix='_g_fov')
        # gaia_summary_stat_BP=calculate_variability_metrics(Time_bp, mag_bp, mag_err_bp, band_suffix='_bp_fov')
        # gaia_summary_stat_RP=calculate_variability_metrics(Time_rp, mag_rp, mag_err_rp, band_suffix='_rp_fov')

        # feats.update(featsG.compute_all_parameters)
        # feats.update(featsBP.compute_all_parameters)
        # feats.update(featsRP.compute_all_parameters)

    return feats


def augment_with_gaia_summary(source_ids: List[int]) -> pd.DataFrame:
    # Placeholder; implement TAP query to gaiadr3 vari_summary and join on source_id

    gaia_params = query_gaia_params(source_ids).to_pandas()
    gaia_summary_stats = query_vari_summary(source_ids).to_pandas()

    cols_to_drop = [
        "solution_id",
        "in_vari_classification_result",
        "in_vari_rrlyrae",
        "in_vari_cepheid",
        "in_vari_planetary_transit",
        "in_vari_short_timescale",
        "in_vari_long_period_variable",
        "in_vari_eclipsing_binary",
        "in_vari_rotation_modulation",
        "in_vari_ms_oscillator",
        "in_vari_agn",
        "in_vari_microlensing",
        "in_vari_compact_companion",
    ]

    gaia_summary_stats = gaia_summary_stats.drop(columns=cols_to_drop)

    gaia_params = gaia_params.set_index("source_id")
    gaia_summary_stats = gaia_summary_stats.set_index("source_id")

    gaia_params = gaia_params.join(gaia_summary_stats, how="left")
    return gaia_params
