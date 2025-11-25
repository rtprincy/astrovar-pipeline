
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
# from .lc_stats import LCStats
from .extract_gaia_parameters import query_gaia_params,query_vari_summary
from .LCStatistics import LCStatistics as lcs
from astroquery.gaia import Gaia

@dataclass
class PeriodogramBundle:
    freq: np.ndarray
    lsp: np.ndarray
    theta: np.ndarray
    psi: np.ndarray

def extract_features_from_lc(lc: pd.DataFrame, per_g: PeriodogramBundle|None=None,per_bp: PeriodogramBundle|None=None,per_rp: PeriodogramBundle|None=None) -> Dict[str, Any]:
    feats = {}
    # Add periodogram features
    if (per_g is not None) & (per_bp is not None) & (per_rp is not None):

        mag_g = lc['g_transit_mag'].values
        Time_g = lc['g_transit_time'].values
        flux_g = lc['g_transit_flux'].values
        flux_err_g = lc['g_transit_flux_error'].values
        mag_err_g = (2.5 / np.log(10)) * (flux_err_g / flux_g)

        mag_bp = lc['bp_mag'].values
        Time_bp = lc['bp_obs_time'].values
        flux_bp = lc['bp_flux'].values
        flux_err_bp = lc['bp_flux_error'].values
        mag_err_bp = (2.5 / np.log(10)) * (flux_err_bp / flux_bp)

        mag_rp = lc['rp_mag'].values
        Time_rp = lc['rp_obs_time'].values
        flux_rp = lc['rp_flux'].values
        flux_err_rp = lc['rp_flux_error'].values
        mag_err_rp = (2.5 / np.log(10)) * (flux_err_rp / flux_rp)

        best_freq_g=per_g.freq[np.nanargmax(per_g.psi)]
        best_freq_bp=per_bp.freq[np.nanargmax(per_bp.psi)]
        best_freq_rp=per_rp.freq[np.nanargmax(per_rp.psi)]

        best_period_g=1/best_freq_g
        best_period_bp=1/best_freq_bp
        best_period_rp=1/best_freq_rp


        std_period=np.std([best_period_g,best_period_bp,best_period_rp])
        frac_period=(best_period_g)/std_period

        print("Extracting features from light curves and periodograms")

        featG=lcs(mag_g, mag_err_g, Time_g, per_g.lsp,per_g.psi, per_g.freq, flux_g, flux_err_g, best_freq_g, 'G')
        featBP=lcs(mag_bp, mag_err_bp, Time_bp, per_bp.lsp,per_bp.psi, per_bp.freq, flux_bp, flux_err_bp, best_freq_bp, 'BP')
        featRP=lcs(mag_rp, mag_err_rp, Time_rp, per_rp.lsp,per_rp.psi, per_rp.freq, flux_rp, flux_err_rp, best_freq_rp, 'RP')

        feats.update(featG.compute_all_parameters())
        feats.update(featBP.compute_all_parameters())
        feats.update(featRP.compute_all_parameters())


        feats.update({"std":std_period,"frac_period":frac_period,"opt_period_g":best_period_g,"opt_period_bp":best_period_bp,"opt_period_rp":best_period_rp})
       
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

    gaia_params=query_gaia_params(source_ids).to_pandas()
    gaia_summary_stats=query_vari_summary(source_ids).to_pandas()

    cols_to_drop=['solution_id','in_vari_classification_result', 'in_vari_rrlyrae', 'in_vari_cepheid',
       'in_vari_planetary_transit', 'in_vari_short_timescale',
       'in_vari_long_period_variable', 'in_vari_eclipsing_binary',
       'in_vari_rotation_modulation', 'in_vari_ms_oscillator', 'in_vari_agn',
       'in_vari_microlensing', 'in_vari_compact_companion']

    gaia_summary_stats=gaia_summary_stats.drop(columns=cols_to_drop)

    gaia_params=gaia_params.set_index("source_id")
    gaia_summary_stats=gaia_summary_stats.set_index("source_id")

    gaia_params = gaia_params.join(gaia_summary_stats, how="left")
    return gaia_params





