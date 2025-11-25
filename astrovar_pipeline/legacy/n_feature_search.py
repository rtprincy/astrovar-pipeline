import numpy as np
import pandas as pd
from openTSNE import TSNE
import time
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import tqdm

from concurrent.futures import ProcessPoolExecutor


import os
import glob



data=pd.read_csv("/idia/users/princy/project_obj_btw_MS_WD/paper3_data/ms_wd_NG_NBP_NRP_abv_25_Gaia_spurious_sources.csv")

data[['amp_G','amp_BP','amp_RP']]=data[['amp_G','amp_BP','amp_RP']].abs()

data['std']=np.std([data['opt_period_G'],data['opt_period_RP'],data['opt_period_BP']],axis=0)

data['frac_period']=data['opt_period_G']/data['std']
all_features=['parallax',
 'parallax_error',
 'phot_g_mean_mag',
 'bp_rp',
 'parallax_over_error',
 'num_selected_g_fov',
 'G_abs',
 'mean_obs_time_g_fov',
 'time_duration_g_fov',
 'min_mag_g_fov',
 'max_mag_g_fov',
 'mean_mag_g_fov',
 'median_mag_g_fov',
 'range_mag_g_fov',
 'trimmed_range_mag_g_fov',
 'std_dev_mag_g_fov',
 'skewness_mag_g_fov',
 'kurtosis_mag_g_fov',
 'mad_mag_g_fov',
 'abbe_mag_g_fov',
 'iqr_mag_g_fov',
 'stetson_mag_g_fov',
 'std_dev_over_rms_err_mag_g_fov',
 'outlier_median_g_fov',
 'num_selected_bp',
 'mean_obs_time_bp',
 'time_duration_bp',
 'min_mag_bp',
 'max_mag_bp',
 'mean_mag_bp',
 'median_mag_bp',
 'range_mag_bp',
 'trimmed_range_mag_bp',
 'std_dev_mag_bp',
 'skewness_mag_bp',
 'kurtosis_mag_bp',
 'mad_mag_bp',
 'abbe_mag_bp',
 'iqr_mag_bp',
 'stetson_mag_bp',
 'std_dev_over_rms_err_mag_bp',
 'outlier_median_bp',
 'num_selected_rp',
 'mean_obs_time_rp',
 'time_duration_rp',
 'min_mag_rp',
 'max_mag_rp',
 'mean_mag_rp',
 'median_mag_rp',
 'range_mag_rp',
 'trimmed_range_mag_rp',
 'std_dev_mag_rp',
 'skewness_mag_rp',
 'kurtosis_mag_rp',
 'mad_mag_rp',
 'abbe_mag_rp',
 'iqr_mag_rp',
 'stetson_mag_rp',
 'std_dev_over_rms_err_mag_rp',
 'outlier_median_rp',
 'opt_period_G',
 'opt_period_BP',
 'opt_period_RP',
 'std',
 'frac_period',
 'log_sigvar',
 'fapG',
 'fapRP',
 'fapBP',
 'amp_G',
 'amp_BP',
 'kurtosisG',
 'p99',
 'p95_100',
 'n05',
 'psi_sigvar',
 'amp_RP',
 'p90_100',
 'p99_100',
 'rms_over_ptp_amp',
'RUWE']

data_ruwe=data[data['RUWE']<=1.4]

features=data_ruwe[all_features]
# features=features[data['in_andromeda_survey']==False]
# features= features.replace([np.inf, -np.inf], np.nan)

features=features.dropna()

correlation_matrix = features.corr().abs()

upper_triangle_mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
upper_triangle = correlation_matrix.where(upper_triangle_mask)

threshold = 0.95
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]


data_reduced = features.drop(columns=to_drop)


# scaler = StandardScaler()

# X_scaled = scaler.fit_transform(data_reduced)


feat_name=data_reduced.columns

indices=np.load("indices_ruwecut_67feat_80perplexity.npy")


# new_cols=list(feat_name[indices[:51]])
    # new_cols=feat_name

print(f"Number of features:{len(feat_name)}")
    
# selected_feat=features[new_cols]
# selected_feat=data_reduced
    
# scaler = StandardScaler()
    
    
# X_scaled_selected = scaler.fit_transform(selected_feat)

# learning_rate=np.arange(50,701,50)  
# i=np.arange(30,101,5)
# i=[80]
i=np.arange(25,len(feat_name),1)

def compute_silhoutte(i):

    new_cols=list(feat_name[indices[:i]])
    # new_cols=feat_name

    print(f"Number of features:{len(new_cols)}")
    
    selected_feat=data_reduced[new_cols]
    
    scaler = StandardScaler()
    
    
    X_scaled_selected = scaler.fit_transform(selected_feat)
    

    tsne_red = TSNE(
        n_components=2,
        perplexity=80,
        initialization="pca",  # Initialize with PCA for stability
        random_state=42,
        # n_jobs=-1,  # Use all CPU cores
        n_iter=1000,
        dof=1
    )
    
    data_tsne_selected = tsne_red.fit(X_scaled_selected)

    gmm = GaussianMixture(n_components=10,covariance_type='tied',n_init=10,random_state=42)
    gmm.fit(data_tsne_selected)
    
    
    labels_gm = gmm.predict(data_tsne_selected)
    silhouette_avg = silhouette_score(data_tsne_selected, labels_gm)


    
    print(f"Silhouette Score: {silhouette_avg}")

    with open("best_nfeatures_ruwecut_80perplexity_silhouette_score.txt",'a') as f:
        f.write("%d %.3f \n"%(i,silhouette_avg))

    # np.save("data_tsne_allfeat_80perplexity_ruwecut.npy",data_tsne_selected)
    # np.save("label_gm_ruwecut_allfeat_80perplexity.npy",labels_gm)

    



with ProcessPoolExecutor() as executor:
    list(tqdm.tqdm(executor.map(compute_silhoutte, i)))




