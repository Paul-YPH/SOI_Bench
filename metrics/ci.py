###############################################
################## Alignment ##################
###############################################

from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

# def compute_ci(adata,
#                batch_key = 'batch',
#                label_key1='Ground Truth',
#                label_key2='Ground Truth',
#                spatial_key='spatial'):
    
#     batch_list = adata.obs[batch_key].unique()
#     adata1 = adata[adata.obs[batch_key] == batch_list[0]]
#     adata2 = adata[adata.obs[batch_key] == batch_list[1]]
    
#     tgt_cor = adata1.obsm[spatial_key]
#     src_cor = adata2.obsm[spatial_key]
    
#     tgt_cell_type = np.array(adata1.obs[label_key1])
#     src_cell_type = np.array(adata2.obs[label_key2])
    
#     kd_tree = cKDTree(src_cor)
#     distances, indices = kd_tree.query(tgt_cor, k=1) 

#     cri = np.mean((tgt_cell_type == src_cell_type[indices])+0)
    
#     return cri

def compute_ci(adata,
               batch_key='batch',
               label_key='Ground Truth',
               spatial_key='spatial'):
    batch_list = sorted(adata.obs[batch_key].unique())
    
    if len(batch_list) < 2:
        raise ValueError("The dataset must contain at least two batches for comparison.")

    results = []

    for i in range(1, len(batch_list)):
        adata1 = adata[adata.obs[batch_key] == batch_list[i-1]]
        adata2 = adata[adata.obs[batch_key] == batch_list[i]]
        
        tgt_cor = adata1.obsm[spatial_key]
        src_cor = adata2.obsm[spatial_key]
        
        tgt_cell_type = np.array(adata1.obs[label_key])
        src_cell_type = np.array(adata2.obs[label_key])
        
        if np.isnan(src_cor).any() or np.isnan(tgt_cor).any():
            raise ValueError("Spatial coordinates contain NaN values.")
        
        kd_tree = cKDTree(src_cor)
        distances, indices = kd_tree.query(tgt_cor, k=1) 

        cri_sum = (tgt_cell_type == src_cell_type[indices]).sum()
        value = cri_sum / len(tgt_cell_type)
        print(f"CRI value for batch {i-1} to {i}: {value}")
        results.append({'metric': 'ci', 'value': value, 'group': f'{batch_list[i-1]}_{batch_list[i]}'})
    df = pd.DataFrame(results)
    return df
