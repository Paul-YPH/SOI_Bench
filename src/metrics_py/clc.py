###############################################
################## Alignment ##################
###############################################

from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

# https://linkinghub.elsevier.com/retrieve/pii/S0092867424011590
def compute_cell_score(i_n, X, Y, mapped_indices, labels1, labels2, nbrs1, nbrs2, K):
    i_prime = mapped_indices[i_n]
    label_consistent = int(labels1[i_n] == labels2[i_prime])
    
    _, neighbors1 = nbrs1.kneighbors([X[i_n]])
    neighbors1 = neighbors1[0]
    mapped_neighbors = [mapped_indices[nbr] for nbr in neighbors1]
    
    _, neighbors2 = nbrs2.kneighbors([Y[i_prime]])
    neighbors2 = neighbors2[0]
    
    spatial_consistency = sum(1 for n_prime in mapped_neighbors if n_prime in neighbors2) / K
    return label_consistent * spatial_consistency

def compute_clc(adata,
                batch_key='batch',
                label_key='Ground Truth',
                spatial_aligned_key='spatial_aligned',
                k_percentage=0.025):
    def ensure_list(data):
        if isinstance(data, (dict, pd.Series)):
            return [data[str(i)] for i in range(len(data))]
        return data
    
    batch_list = adata.uns['config']['sample_list']
    results = []
    pi_list = ensure_list(adata.uns['pi_list'])
    pi_index_list = ensure_list(adata.uns['pi_index_list'])
    pi_column_list = ensure_list(adata.uns['pi_column_list'])
    
    for i in range(len(pi_list)):
        adata1_full = adata[adata.obs[batch_key] == batch_list[i]]
        adata2_full = adata[adata.obs[batch_key] == batch_list[i+1]]

        mask1 = np.isin(pi_index_list[i], adata1_full.obs_names)
        mask2 = np.isin(pi_column_list[i], adata2_full.obs_names)
        
        valid_idx1 = np.array(pi_index_list[i])[mask1]
        valid_idx2 = np.array(pi_column_list[i])[mask2]

        adata1 = adata1_full[valid_idx1]
        adata2 = adata2_full[valid_idx2]

        mapping_matrix = pi_list[i]
        mapping_matrix = mapping_matrix[mask1, :][:, mask2]

        X = np.array(adata1.obsm[spatial_aligned_key])
        Y = np.array(adata2.obsm[spatial_aligned_key])
        
        labels1 = adata1.obs[label_key].values
        labels2 = adata2.obs[label_key].values
        
        N, M = mapping_matrix.shape
        if N == 0 or M == 0: continue 

        avg_cell_count = (N + M) / 2
        theoretical_k = int(np.ceil(k_percentage * avg_cell_count))
        K = max(1, min(theoretical_k, N, M))
        
        mapped_indices = np.argmax(mapping_matrix, axis=1)
        
        nbrs1 = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(X)
        nbrs2 = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(Y)

        cell_scores = Parallel(n_jobs=-1)(
            delayed(compute_cell_score)(
                i_n, X, Y, mapped_indices, labels1, labels2, nbrs1, nbrs2, K
            ) for i_n in range(N)
        )
        clc_score = np.mean(cell_scores)
        results.append({'metric': 'clc', 'value': clc_score, 'group': f'{batch_list[i]}_{batch_list[i+1]}'})
        
    df = pd.DataFrame(results)
    return df