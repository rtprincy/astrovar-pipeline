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
    psi: np.ndarray


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
    # Add periodogram features
    feats = {}
    best_periods = []
    for periodogram, filter in [(per_g, "g"), (per_bp, "bp"), (per_rp, "rp")]:
        lc_x = lc[~lc[f"variability_flag_{filter}_reject"].values]

        ext = "_transit" if filter == "g" else ""
        time_ext = "_obs" if filter != "g" else ""

        mag = lc_x[f"{filter + ext}_mag"].dropna().values
        time = lc_x[f"{filter + ext + time_ext}_time"].dropna().values
        flux = lc_x[f"{filter + ext}_flux"].dropna().values
        flux_err = lc_x[f"{filter + ext}_flux_error"].dropna().values
        mag_err = (2.5 / np.log(10)) * (flux_err / flux)

        best_freq, best_period = optimise_freq(
            oversampling_factor, periodogram.psi, periodogram.freq, time, mag, mag_err
        )
        best_periods.append(best_period)

        featX = lcs(
            filter.upper(),
            mag,
            mag_err,
            time,
            periodogram.lsp,
            periodogram.psi,
            periodogram.freq,
            flux,
            flux_err,
            best_freq,
        )

        feats.update(featX.compute_all_parameters())

    std_period = np.std(best_periods)
    frac_period = best_periods[0] / std_period
    feats.update(
        {
            "std": std_period,
            "frac_period": frac_period,
            "opt_period_g": best_periods[0],
            "opt_period_bp": best_periods[1],
            "opt_period_rp": best_periods[2],
        }
    )

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
