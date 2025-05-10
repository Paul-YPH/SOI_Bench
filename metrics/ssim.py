###############################################
################## Alignment ##################
###############################################

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .utils import extract_exp, extract_hvg

def compute_ssim(adata,
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
            # No grid mode, process the data directly
            
            # Extract expression data for HVGs
            tgt_exp = extract_exp(adata1, layer=layer, dataframe=False, gene=extract_hvg(adata1, adata2))
            src_exp = extract_exp(adata2, layer=layer, dataframe=False, gene=extract_hvg(adata1, adata2))
            
            # Extract spatial coordinates
            tgt_cor = adata1.obsm[spatial_key]
            src_cor = adata2.obsm[spatial_key]
            
            # Find nearest neighbors
            kd_tree = cKDTree(src_cor)
            distances, indices = kd_tree.query(tgt_cor, k=1)
            
            # Compute SSIM
            ssim_values = []
            for i in range(tgt_exp.shape[0]):
                row_ssim = ssim_function(tgt_exp[i], src_exp[indices[i]])
                ssim_values.append(row_ssim)
            ssim = np.mean(ssim_values)
            
            results.append({'metric': 'ssim_pair', 'value': ssim, 'group': f'{batch_list[batch_idx]}_{batch_list[batch_idx+1]}'})
            
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
            
            grid_ssims = []
            
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
                        
                        corr = ssim_function(tgt_vector, src_vector)
                        
                        if not np.isnan(corr):
                            grid_ssims.append(corr)
                        
            # Compute average SSIM across all grids
            ssim = np.mean(grid_ssims)
        
            results.append({'metric': 'ssim_grid', 'value': ssim, 'group': f'{batch_list[batch_idx]}_{batch_list[batch_idx+1]}'})
            
    df = pd.DataFrame(results)
    return df

def ssim_function(im1,im2,M=1):
    im1, im2 = im1/im1.max(), im2/im2.max()
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim