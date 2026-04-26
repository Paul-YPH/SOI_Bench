###############################################
################## Alignment ##################
###############################################

from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

def compute_ci(adata,
               batch_key='batch',
               label_key='Ground Truth',
               spatial_aligned_key='spatial_aligned'):
    
    batch_list = adata.uns['config']['sample_list']

    results = []

    for i in range(1, len(batch_list)):
        adata1 = adata[adata.obs[batch_key] == batch_list[i-1]]
        adata2 = adata[adata.obs[batch_key] == batch_list[i]]
        
        tgt_cor = adata1.obsm[spatial_aligned_key]
        src_cor = adata2.obsm[spatial_aligned_key]
        
        tgt_cell_type = np.array(adata1.obs[label_key])
        src_cell_type = np.array(adata2.obs[label_key])
        
        if np.isnan(src_cor).any() or np.isnan(tgt_cor).any():
            raise ValueError("Spatial coordinates contain NaN values.")
        
        kd_tree = cKDTree(src_cor)
        distances, indices = kd_tree.query(tgt_cor, k=1) 

        cri_sum = (tgt_cell_type == src_cell_type[indices]).sum()
        value = cri_sum / len(tgt_cell_type)
        #print(f"CRI value for batch {i-1} to {i}: {value}")
        results.append({'metric': 'ci', 'value': value, 'group': f'{batch_list[i-1]}_{batch_list[i]}'})
    df = pd.DataFrame(results)
    return df
