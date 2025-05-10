#########################################################
################## Alignment/Embedding ##################
#########################################################

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
                  label_key = None,
                  cluster_key = None,
                  batch_key = None,
                  spatial_key = 'spatial'):

    if cluster_key is not None:
        key = cluster_key   
        results = []
        for batch in adata.obs[batch_key].unique():
            label_batch = adata.obs[key][adata.obs[batch_key] == batch]
            location_batch = adata.obsm[spatial_key][adata.obs[batch_key] == batch]
            chaos_value = _compute_chaos(label_batch, location_batch)
            results.append({'metric': 'chaos', 'value': chaos_value, 'group': f'{batch}'})    
        df = pd.DataFrame(results)
    elif label_key is not None:
        key = label_key
        label = adata.obs[key]
        location = adata.obsm[spatial_key]
        chaos_value = _compute_chaos(label, location)
        df = pd.DataFrame({'metric': 'chaos', 'value': [chaos_value], 'group': [key]})
    else:
        raise ValueError("Either `cluster_key` or `label_key` must be provided.")
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