#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from openTSNE import TSNE
import matplotlib.pyplot as plt
import time
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
# import umap
import tqdm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans

params = {
         'mathtext.default': 'regular',

          'text.usetex': True}
plt.rcParams.update(params)

plt.rcParams['figure.dpi']=250
plt.rcParams['lines.color']='k'
plt.rcParams['axes.edgecolor']='k'
# plt.rcParams['lines.linewidth']=1
# plt.rcParams['lines.markeredgewidth']=1
plt.rcParams['xtick.minor.visible']=False
plt.rcParams['ytick.minor.visible']=False
plt.rcParams['axes.labelsize']=22
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18

save_ls_to="/idia/users/princy/gaia_periodogram/"

import os
import glob
import matplotlib.ticker as ticker


# In[2]:


pulsating_hsd=pd.read_csv("pulsating_hsd_known_mode_Krzesinski2022.csv")
CVs_lit=pd.read_csv("CVs_cluster_with_known_CVs.csv")
pmode_sdb=pd.read_csv("pmode_sdB_Baran2024_tsnecomponents.csv")


# In[3]:


data=pd.read_csv("/idia/users/princy/project_obj_btw_MS_WD/paper3_data/ms_wd_NG_NBP_NRP_abv_25_Gaia_spurious_sources.csv")


# In[5]:


data[['amp_G','amp_BP','amp_RP']]=data[['amp_G','amp_BP','amp_RP']].abs()
data['std']=np.std([data['opt_period_G'],data['opt_period_RP'],data['opt_period_BP']],axis=0)
data['frac_period']=data['opt_period_G']/data['std']


# In[6]:


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


features=data[all_features]

features=features.dropna()

correlation_matrix = features.corr().abs()

upper_triangle_mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
upper_triangle = correlation_matrix.where(upper_triangle_mask)

threshold = 0.95
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]


data_reduced = features.drop(columns=to_drop)


scaler = StandardScaler()

X_scaled = scaler.fit_transform(data_reduced)



# In[6]:


X_scaled.shape


# In[18]:


tsne = TSNE(
    n_components=2,
    perplexity=70,
    initialization="pca",
    random_state=42,
    n_iter=1000,
    dof=1
)

data_tsne = tsne.fit(X_scaled)
# data_tsne=np.load("data_tsne_66feat_perplexity70_fulldata.npy")


# In[16]:


fig, ax = plt.subplots(figsize=(6, 5.5))

ax.scatter(data_tsne[:, 0], data_tsne[:, 1],s=1,color='grey')

ax.set_title("t-SNE with 66 features",fontsize=20)
ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')

plt.tight_layout()

# plt.savefig("tsne_components_66features_plain.png",format='png')



color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# In[25]:


gmm = GaussianMixture(n_components=10,covariance_type='tied',n_init=10,random_state=42)
gmm.fit(data_tsne)



labels_gm = gmm.predict(data_tsne)
silhouette_avg = silhouette_score(data_tsne, labels_gm)
calinski_harabasz = calinski_harabasz_score(data_tsne, labels_gm)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Calinski-Harabasz Score: {calinski_harabasz}")


plt.figure(figsize=(8, 6))
unique_labels = np.unique(labels_gm)


cmap = plt.get_cmap('viridis')
markers=['o','s','^','*','>','<','+','D','o','s','<','>']

i=0
for label in unique_labels:

    label_indices = labels_gm == label
    plt.scatter(data_tsne[label_indices, 0], data_tsne[label_indices, 1], 
                label=f'Cluster {label}', 
                c=color[i], 
                s=10, marker=markers[i])
    i+=1


plt.xlabel('tSNE Component 1')
plt.ylabel('tSNE Component 2')


plt.legend()



feat_name=data_reduced.columns

rf = RandomForestClassifier(n_estimators=100, criterion='log_loss',random_state=0)
rf.fit(X_scaled, labels_gm)




importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]


print("Feature ranking:")

for i in range(X_scaled.shape[1]):
    print(f"{i + 1}. {feat_name[indices[i]]} ({importances[indices[i]]:.4f})")


plt.figure(figsize=(15, 6))
plt.title("Feature Importances")
plt.bar(range(X_scaled.shape[1]), importances[indices], align='center')
plt.xticks(range(X_scaled.shape[1]), indices)
plt.xlim([-1, X_scaled.shape[1]])
plt.xlabel('Feature Index')
plt.ylabel('Importance')



new_cols=list(feat_name[indices[:51]])

selected_feat=features[new_cols]

scaler = StandardScaler()


X_scaled_selected = scaler.fit_transform(selected_feat)


# In[14]:


tsne_red = TSNE(
    n_components=2,
    perplexity=70,
    initialization="pca",  # Initialize with PCA for stability
    random_state=42,
    # n_jobs=-1,  # Use all CPU cores
    n_iter=1000,
    dof=1
)

data_tsne_selected = tsne_red.fit(X_scaled_selected)



fig, ax = plt.subplots(figsize=(9, 8.5))

ax.scatter(data_tsne_selected[:, 0], data_tsne_selected[:, 1],s=1,color='grey')
# ax.scatter(data_tsne[:, 0][mask], data_tsne[:, 1][mask],s=1,color='blue')

ax.set_title("t-SNE with 51 features",fontsize=35)
ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
# plt.scatter(data_tsne[:,0][data['C0']==True],data_tsne[:,1][data['C0']==True],s=3,label='cluster 0')
# plt.scatter(data_tsne[:,0][data['C1']==True],data_tsne[:,1][data['C1']==True],s=3,label='cluster 1')
# plt.scatter(data_tsne[:,0][data['C2']==True],data_tsne[:,1][data['C2']==True],s=1, label='cluster 2')
# plt.scatter(data_tsne[:,0][data['hsd']==True],data_tsne[:,1][data['hsd']==True],s=15,facecolor='white',edgecolor=color[0],marker='o', label='Hot sd')
# plt.scatter(data_tsne[:,0][data['CVs']==True],data_tsne[:,1][data['CVs']==True],s=15,facecolor='white',edgecolor=color[3],marker='<', label='CVs')
# plt.scatter(data_tsne[:,0][data['WD_simbad']==True],data_tsne[:,1][data['WD_simbad']==True],s=12,facecolor='white',edgecolor=color[1],marker='s',label='WD')
# plt.scatter(data_tsne[:,0][data['in_andromeda_survey']==True],data_tsne[:,1][data['in_andromeda_survey']==True],s=3, label='GAPS sources')

# plt.scatter(data_tsne_red[:, 0], data_tsne_red[:, 1],s=1,c=np.log10(data_temp['opt_period_G'].values),cmap='viridis')
# plt.colorbar(label='period')
# plt.scatter(tsne_pt[:, 0], tsne_pt[:, 1],s=10)
# ax.legend(loc='best',fontsize=12,markerscale=3)
ax.set_xlabel('t-SNE Component 1',fontsize=40)
ax.set_ylabel('t-SNE Component 2',fontsize=40)
ax.set_ylim([-100,90])
ax.set_xlim([-90,100])
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

for spine in ax.spines.values():
    spine.set_edgecolor('black')   # Set color
    spine.set_linewidth(1.4)  

plt.tight_layout()

plt.savefig("tsne_components_51features_perplexity70_all_data.pdf",format='pdf')

