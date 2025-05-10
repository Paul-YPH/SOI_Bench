###############################################
################## Alignment ##################
###############################################

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .utils import extract_exp, extract_hvg

def compute_pcc(adata,
                batch_key = 'batch',
                label_key='Ground Truth',
                layer=None,
                spatial_key='spatial',
                grid_num=None):

    batch_list = adata.obs[batch_key].unique()
    
    results = []
    for batch_idx in range(len(batch_list)-1):
        adata1 = adata[adata.obs[batch_key] == batch_list[batch_idx]]
        adata2 = adata[adata.obs[batch_key] == batch_list[batch_idx+1]]

        if grid_num is None:
            # Extract expression data for HVGs
            hvg = extract_hvg(adata1, adata2)
            tgt_exp = extract_exp(adata1, layer=layer, dataframe=False, gene=hvg)
            src_exp = extract_exp(adata2, layer=layer, dataframe=False, gene=hvg)
            
            # Extract spatial coordinates
            tgt_cor = adata1.obsm[spatial_key]
            src_cor = adata2.obsm[spatial_key]
            
            # Find nearest neighbors
            kd_tree = cKDTree(src_cor)
            distances, indices = kd_tree.query(tgt_cor, k=1)
            
            # Compute PCC
            src_exp_nn = src_exp[indices]
            A = tgt_exp.astype(np.float64)  
            B = src_exp_nn.astype(np.float64)
            A_mean = A.mean(axis=1, keepdims=True)
            B_mean = B.mean(axis=1, keepdims=True)
            A_centered = A - A_mean
            B_centered = B - B_mean
            numerator = np.sum(A_centered * B_centered, axis=1)
            denom = np.sqrt(np.sum(A_centered**2, axis=1) * np.sum(B_centered**2, axis=1))
            pcc_values = numerator / denom
            pcc = pcc_values.mean()
            
            results.append({'metric': 'pcc_pair', 'value': pcc, 'group': f'{batch_list[batch_idx]}_{batch_list[batch_idx+1]}'})
        
        else:
            # Grid mode
            print(f"Processing with grid_num={grid_num}...")

            # Extract spatial coordinates
            tgt_cor = adata1.obsm[spatial_key]
            src_cor = adata2.obsm[spatial_key]
            
            # Determine grid bounds
            xmin, ymin = min(tgt_cor[:, 0].min(), src_cor[:, 0].min()), min(tgt_cor[:, 1].min(), src_cor[:, 1].min())
            xmax, ymax = max(tgt_cor[:, 0].max(), src_cor[:, 0].max()), max(tgt_cor[:, 1].max(), src_cor[:, 1].max())
            
            # Create grid intervals
            x_intervals = np.linspace(xmin, xmax, grid_num + 1)
            y_intervals = np.linspace(ymin, ymax, grid_num + 1)
            
            grid_pccs = []
            
            # Iterate through each grid
            for i in range(grid_num):
                for j in range(grid_num):
                    # Define grid boundaries
                    x_min, x_max = x_intervals[i], x_intervals[i + 1]
                    y_min, y_max = y_intervals[j], y_intervals[j + 1]
                    
                    # Select cells in this grid for both datasets
                    tgt_indices = (tgt_cor[:, 0] >= x_min) & (tgt_cor[:, 0] < x_max) & \
                                (tgt_cor[:, 1] >= y_min) & (tgt_cor[:, 1] < y_max)
                    src_indices = (src_cor[:, 0] >= x_min) & (src_cor[:, 0] < x_max) & \
                                (src_cor[:, 1] >= y_min) & (src_cor[:, 1] < y_max)
                    
                    if np.any(tgt_indices) and np.any(src_indices):
                        
                        tgt_cell_types = adata1.obs[label_key][tgt_indices]
                        src_cell_types = adata2.obs[label_key][src_indices]
                        
                        tgt_type_counts = tgt_cell_types.value_counts(normalize=True).sort_index()
                        src_type_counts = src_cell_types.value_counts(normalize=True).sort_index()
                        
                        all_types = sorted(set(tgt_type_counts.index).union(set(src_type_counts.index)))
                        tgt_vector = tgt_type_counts.reindex(all_types, fill_value=0).values
                        src_vector = src_type_counts.reindex(all_types, fill_value=0).values
                        
                        corr = np.corrcoef(tgt_vector, src_vector)[0, 1]
                        
                        if not np.isnan(corr):
                            grid_pccs.append(corr)
                        
            # Compute average PCC across all grids
            pcc = np.mean(grid_pccs)

            results.append({'metric': 'pcc_grid', 'value': pcc, 'group': f'{batch_list[batch_idx]}_{batch_list[batch_idx+1]}'})
            
    df = pd.DataFrame(results)
    return df