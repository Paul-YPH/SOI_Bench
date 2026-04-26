#####################################################
################## Spatial Pattern ##################
#####################################################

from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
import numpy as np
import pandas as pd

# https://github.com/zhaofangyuan98/SDMBench/blob/d60653469380f6bd0f574b732c0a20f8a984aafc/SDMBench/SDMBench.py

def fx_1NN(i,location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
    dist_array[i] = np.inf
    return np.min(dist_array)
    
def compute_chaos(adata, 
                  label_key = 'Ground Truth',
                  cluster_key = 'benchmark_cluster',
                  batch_key = 'batch',
                  spatial_key = 'spatial',
                  spatial_aligned_key = 'spatial_aligned'):
    results = []
    if cluster_key in adata.obs.columns:
        for batch in adata.obs[batch_key].unique():
            label_batch = adata.obs[cluster_key][adata.obs[batch_key] == batch]
            location_batch = adata.obsm[spatial_key][adata.obs[batch_key] == batch]
            chaos_value = _compute_chaos(label_batch, location_batch)
            results.append({'metric': 'chaos', 'value': chaos_value, 'group': f'{batch}'})    
    if spatial_aligned_key in adata.obsm.keys():
        label = adata.obs[label_key]
        location = adata.obsm[spatial_aligned_key]

        max_cells = 50000
        if location.shape[0] > max_cells:
            np.random.seed(42) 
            sub_idx = np.random.choice(location.shape[0], max_cells, replace=False)
            if isinstance(label, (pd.Series, pd.Index)):
                label = label.iloc[sub_idx]
            else:
                label = label[sub_idx]
            location = location[sub_idx]

        chaos_value = _compute_chaos(label, location)
        results.append({'metric': 'chaos', 'value': chaos_value, 'group': f'{label_key}'})
    df = pd.DataFrame(results)
    return df

def _compute_chaos(clusterlabel, location):

    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel==k,:]
        if len(location_cluster)<=2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i,location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val)/len(clusterlabel)