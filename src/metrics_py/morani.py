#####################################################
################## Spatial Pattern ##################
#####################################################

import pandas as pd
from pandas import get_dummies
from sklearn.neighbors import kneighbors_graph
import scanpy as sc
import numpy as np

def compute_moran_I(adata, 
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
            moran_I_value = _compute_moranI(label_batch, location_batch)
            results.append({'metric': 'moran_I', 'value': moran_I_value, 'group': f'{batch}'})
    if spatial_aligned_key in adata.obsm.keys():
        label = adata.obs[label_key]
        location = adata.obsm[spatial_aligned_key]
        moran_I_value = _compute_moranI(label, location)
        results.append({'metric': 'moran_I', 'value': moran_I_value, 'group': f'{label_key}'})
    df = pd.DataFrame(results)
    return df

def _compute_moranI(label, location):
    # sc.pp.neighbors(adata, use_rep='spatial')
    g = kneighbors_graph(location, 6, mode='connectivity', metric='euclidean')
    one_hot = get_dummies(label)
    
    moranI = sc.metrics.morans_i(g, one_hot.values.T)
    moranI = np.nanmean(moranI)
    if np.isnan(moranI):
            moranI = 0.0
    return moranI
