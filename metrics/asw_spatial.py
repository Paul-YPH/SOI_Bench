from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd

def compute_asw_spatial(adata, 
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
            asw_spatail_value = _compute_asw_spatial(label_batch, location_batch)
            results.append({'metric': 'asw_spatial', 'value': asw_spatail_value, 'group': f'{batch}'})    
        df = pd.DataFrame(results)
    elif label_key is not None:
        key = label_key
        label = adata.obs[key]
        location = adata.obsm[spatial_key]
        asw_spatail_value = _compute_asw_spatial(label, location)
        df = pd.DataFrame({'metric': 'asw_spatial', 'value': [asw_spatail_value], 'group': [key]})
    else:
        raise ValueError("Either `cluster_key` or `label_key` must be provided.")
    return df

def _compute_asw_spatial(clusterlabel, location):
    if len(np.unique(clusterlabel)) < 2:
        return -1.0  # bad
    d = squareform(pdist(location))
    return silhouette_score(X=d, labels=clusterlabel, metric='precomputed')