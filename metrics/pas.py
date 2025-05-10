#########################################################
################## Alignment/Embedding ##################
#########################################################

from scipy.spatial import distance_matrix
import numpy as np
import pandas as pd
# https://github.com/zhaofangyuan98/SDMBench/blob/d60653469380f6bd0f574b732c0a20f8a984aafc/SDMBench/SDMBench.py
def compute_pas(adata, 
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
            pas_value = _compute_pas_optimized(label_batch, location_batch)
            results.append({'metric': 'pas', 'value': pas_value, 'group': f'{batch}'})    
        df = pd.DataFrame(results)
    elif label_key is not None:
        key = label_key
        label = adata.obs[key]
        location = adata.obsm[spatial_key]
        pas_value = _compute_pas_optimized(label, location)
        df = pd.DataFrame({'metric': 'pas', 'value': [pas_value], 'group': [key]})
    else:
        raise ValueError("Either `cluster_key` or `label_key` must be provided.")
    return df

from scipy.spatial import cKDTree
def _compute_pas_optimized(clusterlabel, location, k=10):
    tree = cKDTree(location)
    distances, indices = tree.query(location, k=k+1)
    neighbor_indices = indices[:, 1:]
    clusterlabel = np.array(clusterlabel)
    neighbor_clusters = clusterlabel[neighbor_indices]
    mismatches = (neighbor_clusters != clusterlabel[:, None]).sum(axis=1)
    results = (mismatches > (k / 2)).astype(np.float64)
    return results.mean()

# def fx_kNN(i,location_in,k,cluster_in):

#     location_in = np.array(location_in)
#     cluster_in = np.array(cluster_in)

#     dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
#     dist_array[i] = np.inf
#     ind = np.argsort(dist_array)[:k]
#     cluster_use = np.array(cluster_in)
#     if np.sum(cluster_use[ind]!=cluster_in[i])>(k/2):
#         return 1
#     else:
#         return 0
    
# def _compute_pas(clusterlabel,location):
    
#     clusterlabel = np.array(clusterlabel)
#     location = np.array(location)
#     matched_location = location
#     results = [fx_kNN(i,matched_location,k=10,cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
#     return np.sum(results)/len(clusterlabel)