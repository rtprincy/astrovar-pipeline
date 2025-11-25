import os
import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from lk_stat_package import lk_stat


def freq_grid(times,oversample_factor=10,f0=None,fn=None):
    times=np.sort(times)
    df = 1.0 / (times.max() - times.min())
    if f0 is None:
        f0 = df
    if fn is None:
        fn = 0.5 / np.median(np.diff(times)) 
    return np.arange(f0, fn, df / oversample_factor)


# Parallel CSV reading function
def read_csv_parallel(file_path):
    return pd.read_csv(file_path)


# Process a single source ID
def process_source(source_id):
    file_path = f'/idia/users/princy/project_obj_btw_MS_WD/gaia_lightcurves/{source_id}.csv'
    save_ls_to="/idia/users/princy/project_obj_btw_MS_WD/gaia_periodograms/"
    theta_path = save_ls_to + 'theta/' + f'{source_id}_{passbands}.npy'
    
    if os.path.exists(file_path) and not os.path.exists(theta_path):
        
        df = read_csv_parallel(file_path)
        df = df[~df['variability_flag_g_reject']]  # Filtering
    
        if df.shape[0] >= 24:
            
            mag = df['g_transit_mag'].dropna().values
            Time = df['g_transit_time'].dropna().values

            flux = df['g_transit_flux'].dropna().values
            flux_err = df['g_transit_flux_error'].dropna().values


            frequencies = freq_grid(Time, oversample_factor=10, f0=None, fn=360)
            periods = 1 / frequencies
            mag_err = (2.5 / np.log(10)) * (flux_err / flux)
          
            theta = lk_stat(periods, mag, mag_err, Time)
            
            np.save(theta_path, theta)


passbands = 'G'
overwrite=False
data=pd.read_csv("/idia/users/princy/project_obj_btw_MS_WD/paper3_data/ms_wd_targets_summary_stat_N_G_BP_RP_25.csv")

ids=data['source_id'].values

new_ids=[]

for source_id in tqdm.tqdm(ids):
    path='/idia/users/princy/project_obj_btw_MS_WD/gaia_periodograms/theta/%d_%s.npy'%(source_id,passbands)

    if os.path.exists(path)==False:
        new_ids.append(source_id)



with ProcessPoolExecutor() as executor:
    list(tqdm.tqdm(executor.map(process_source, new_ids)))
