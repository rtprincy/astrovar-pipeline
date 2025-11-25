import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from astropy.io import fits, ascii
from astropy.table import Table, vstack
import tqdm
import scipy.stats as stats
from utils import *
from astropy.timeseries import LombScargle
import os


from concurrent.futures import ProcessPoolExecutor

import LCStatistics as lcs

data0=pd.read_csv("project_obj_btw_MS_WD/paper3_data/ms_wd_targets_summary_stat_N_G_BP_RP_25.csv")

G=pd.read_csv("project_obj_btw_MS_WD/paper3_data/ms_wd_targets_summary_stat_N_G_BP_RP_25_G.csv",
             usecols=['opt_period_G'])
BP=pd.read_csv("project_obj_btw_MS_WD/paper3_data/ms_wd_targets_summary_stat_N_G_BP_RP_25_BP.csv",
             usecols=['opt_period_BP'])
RP=pd.read_csv("project_obj_btw_MS_WD/paper3_data/ms_wd_targets_summary_stat_N_G_BP_RP_25_RP.csv",
             usecols=['opt_period_RP'])

merged=pd.concat([data0,G,BP,RP],axis=1)

merged['std']=merged[['opt_period_G','opt_period_BP','opt_period_RP']].std(axis=1)

merged['frac_period']=merged['opt_period_G']/merged['std']

merged.dropna(inplace=True)

merged.index=np.arange(len(merged))

ids=merged['source_id'].values


save_ls_to="project_obj_btw_MS_WD/gaia_periodograms/"

lsp_path="project_obj_btw_MS_WD/gaia_periodograms/lsp/"
theta_path="project_obj_btw_MS_WD/gaia_periodograms/theta/"

cols=['log_sigvar','fapG','fapRP','fapBP',
    'amp_G','amp_BP','kurtosisG','p99','p95_100','n05','psi_sigvar','amp_RP',
     'p90_100','p99_100','rms_over_ptp_amp']


features_frame = pd.DataFrame(np.nan, index=range(len(merged)), columns=cols)

def process_source(source_id):

    lsp_G_path=lsp_path+str(source_id)+'_G.npy'
    theta_G_path=theta_path+str(source_id)+'_G.npy'
    
    
    lsp_rp_path=lsp_path+str(source_id)+'_RP.npy'
    theta_rp_path= theta_path+str(source_id)+'_RP.npy'
    
    
    lsp_bp_path=lsp_path+str(source_id)+'_BP.npy'
    theta_bp_path=theta_path+str(source_id)+'_BP.npy'

    file_path = f'project_obj_btw_MS_WD/gaia_lightcurves/{source_id}.csv'
        
    df_ = pd.read_csv(file_path)

    ####################### G Band ###############################
    df = df_[~df_['variability_flag_g_reject']] 
    
    mag = df['g_transit_mag'].dropna().values
    Time = df['g_transit_time'].dropna().values

    flux = df['g_transit_flux'].dropna().values
    flux_err = df['g_transit_flux_error'].dropna().values

    mag_err = (2.5 / np.log(10)) * (flux_err / flux)

    periodG=merged['opt_period_G'][merged['source_id']==source_id].values[0]
    
    best_freq_G=1/periodG
    
    freq_G,lsp_G=np.load(lsp_G_path)
    theta_G=np.load(theta_G_path)
        
    psi_G=(2*lsp_G)/theta_G

    featG=lcs.LCStatistics(mag, mag_err, Time, lsp_G,psi_G, freq_G, flux, flux_err, best_freq_G)
    log_sigvar=featG.log_sigvar()
    fapG=featG.fap()
    amp_G=featG.amplitude()
    kurtosisG=featG.Kurtosis()
    p99=featG.p99()
    p95_100=featG.p90_95_99()[1]
    n05=featG.n05()
    psi_sigvar=featG.psi_sigvar()
    p90_100=featG.p90_95_99()[0]
    p99_100=featG.p90_95_99()[2]
    rms_over_ptp_amp=featG.rms_over_ptp_amp()

    ####################### RP Band ###############################
    dfRP = df_[~df_['variability_flag_rp_reject']] 

    mag_rp = dfRP['rp_mag'].dropna().values
    Time_rp = dfRP['rp_obs_time'].dropna().values

    flux_rp = dfRP['rp_flux'].dropna().values
    flux_err_rp = dfRP['rp_flux_error'].dropna().values

    mag_err_rp = (2.5 / np.log(10)) * (flux_err_rp / flux_rp)

    periodRP=merged['opt_period_RP'][merged['source_id']==source_id].values[0]
    
    best_freq_RP=1/periodRP
    
    freq_RP,lsp_RP=np.load(lsp_rp_path)
    theta_RP=np.load(theta_rp_path)
        
    psi_RP=(2*lsp_RP)/theta_RP

    featRP=lcs.LCStatistics(mag_rp, mag_err_rp, Time_rp,lsp_RP, psi_RP, freq_RP, flux_rp, flux_err_rp, best_freq_RP)

    fapRP=featRP.fap()
    amp_RP=featRP.amplitude()

    ####################### BP Band ###############################
    
    dfBP = df_[~df_['variability_flag_bp_reject']] 

    mag_bp = dfBP['bp_mag'].dropna().values
    Time_bp = dfBP['bp_obs_time'].dropna().values

    flux_bp = dfBP['bp_flux'].dropna().values
    flux_err_bp = dfBP['bp_flux_error'].dropna().values

    mag_err_bp = (2.5 / np.log(10)) * (flux_err_bp / flux_bp)

    periodBP=merged['opt_period_BP'][merged['source_id']==source_id].values[0]
    
    best_freq_BP=1/periodBP
    
    freq_BP,lsp_BP=np.load(lsp_bp_path)
    theta_BP=np.load(theta_bp_path)
        
    psi_BP=(2*lsp_BP)/theta_BP

    featBP=lcs.LCStatistics(mag_bp, mag_err_bp, Time_bp, lsp_BP,psi_BP, freq_BP, flux_bp, flux_err_bp, best_freq_BP)

    fapBP=featBP.fap()
    amp_BP=featBP.amplitude()


    feature_row=[log_sigvar,fapG,fapRP,fapBP,
    amp_G,amp_BP,kurtosisG,p99,p95_100,n05,psi_sigvar,amp_RP,
     p90_100,p99_100,rms_over_ptp_amp]

    

    return feature_row

    
features=[]
with ProcessPoolExecutor() as executor:

    for feature_row in tqdm.tqdm(executor.map(process_source, ids), total=len(ids)):
        features.append(feature_row)
       

merged[cols]=pd.DataFrame(features,columns=cols)

merged.to_csv("ms_wd_targets_summary_stat_N_G_BP_RP_25_features.csv",index=None)
