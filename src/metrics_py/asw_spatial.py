#####################################################
################## Spatial Pattern ##################
#####################################################

from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd

def compute_asw_spatial(adata, 
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
            asw_spatail_value = _compute_asw_spatial(label_batch, location_batch)
            results.append({'metric': 'asw_spatial', 'value': asw_spatail_value, 'group': f'{batch}'})    
    if spatial_aligned_key in adata.obsm.keys():
        label = adata.obs[label_key]
        location = adata.obsm[spatial_aligned_key]
        asw_spatail_value = _compute_asw_spatial(label, location)
        results.append({'metric': 'asw_spatial', 'value': asw_spatail_value, 'group': f'{label_key}'})
    df = pd.DataFrame(results)
    return df

def _compute_asw_spatial(clusterlabel, location):
    if len(np.unique(clusterlabel)) < 2:
        return -1.0  # bad
    d = squareform(pdist(location))
    return silhouette_score(X=d, labels=clusterlabel, metric='precomputed')