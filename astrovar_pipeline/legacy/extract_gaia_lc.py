from astroquery.gaia import Gaia

from pathlib import Path
import os
import pandas as pd
import tqdm

Gaia.login(user='username', password='***********')

data=pd.read_csv("/idia/users/princy/project_obj_btw_MS_WD/paper3_data/ms_wd_targets_summary_stat.csv")


data=data[(data['num_selected_g_fov']>24)&(data['num_selected_bp']>24)&(data['num_selected_rp']>24)]
ids=data['source_id'].values

new_ids=[]

for source_id in tqdm.tqdm(ids):
    path='/idia/users/princy/project_obj_btw_MS_WD/gaia_lightcurves/%d.csv'%(source_id)

    if os.path.exists(path)==False:
        new_ids.append(source_id)

for source_id in tqdm.tqdm(new_ids):
    path='/idia/users/princy/project_obj_btw_MS_WD/gaia_lightcurves/%d.csv'%(source_id)

    lc=Gaia.load_data([source_id],data_release='Gaia DR3',data_structure='INDIVIDUAL',retrieval_type='EPOCH_PHOTOMETRY',valid_data=True,format='fits'
                             )
    gaia_lc=lc[list(lc.keys())[0]][0]
    gband_frame=gaia_lc.to_pandas()
    gband_frame.to_csv(path,index=None)





    

