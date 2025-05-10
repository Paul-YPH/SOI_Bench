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
                pi=None, 
                spatial_key='spatial',
                k_percentage=0.025):
    batch_list = adata.obs[batch_key].unique()
    results = []

    for b in range(len(batch_list)-1):
        adata1 = adata[adata.obs[batch_key] == batch_list[b]]
        adata2 = adata[adata.obs[batch_key] == batch_list[b+1]]

        print(f"mapping_matrix shape: {pi[b].shape}")
        print(f"adata1 shape: {adata1.shape}, adata2 shape: {adata2.shape}")

        adata1 = adata1[pi[b].index]
        adata2 = adata2[pi[b].columns]

        mapping_matrix = pi[b].values
        
        X = np.array(adata1.obsm[spatial_key])
        Y = np.array(adata2.obsm[spatial_key])
        
        labels1 = adata1.obs[label_key].values
        labels2 = adata2.obs[label_key].values
        
        N, M = mapping_matrix.shape

        avg_cell_count = (N + M) / 2
        K = max(1, int(np.ceil(k_percentage * avg_cell_count)))
        
        mapped_indices = np.argmax(mapping_matrix, axis=1)
        
        nbrs1 = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(X)
        nbrs2 = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(Y)

        cell_scores = Parallel(n_jobs=-1)(
            delayed(compute_cell_score)(
                i_n, X, Y, mapped_indices, labels1, labels2, nbrs1, nbrs2, K
            ) for i_n in range(N)
        )
        clc_score = np.mean(cell_scores)
        results.append({'metric': 'clc', 'value': clc_score, 'group': f'{batch_list[b]}_{batch_list[b+1]}'})
        
    df = pd.DataFrame(results)
    return df

# def compute_clc(adata,
#                   batch_key='batch',
#                   label_key='cluster',
#                   pi=None, 
#                   spatial_key='spatial',
#                   k_percentage=0.025):
#     batch_list = adata.obs[batch_key].unique()
#     results = []
    
#     for i in range(len(batch_list)-1):
#         adata1 = adata[adata.obs[batch_key] == batch_list[i]]
#         adata2 = adata[adata.obs[batch_key] == batch_list[i+1]]
        
#         mapping_matrix = pi[i].values
        
#         X = np.array(adata1.obsm[spatial_key])
#         Y = np.array(adata2.obsm[spatial_key])
        
#         labels1 = adata1.obs[label_key].values
#         labels2 = adata2.obs[label_key].values
        
#         # Number of cells in each dataset
#         N, M = mapping_matrix.shape

#         # Determine K for nearest neighbors
#         avg_cell_count = (N + M) / 2
#         K = max(1, int(np.ceil(k_percentage * avg_cell_count)))

#         # Find the most probable mapping for each cell in dataset 1
#         mapped_indices = np.argmax(mapping_matrix, axis=0)
#         # Build nearest neighbors for both datasets
#         nbrs1 = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(X)
#         nbrs2 = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(Y)

#         # Calculate CLC score
#         clc_score = 0
#         for i_n in range(N):
#             # Get the mapped index in dataset 2
#             i_prime = mapped_indices[i_n]

#             # Check label consistency
#             label_consistent = int(labels1[i_n] == labels2[i_prime])

#             # Get neighbors of cell i in dataset 1
#             _, neighbors1 = nbrs1.kneighbors([X[i_n]])
#             neighbors1 = neighbors1[0]

#             # Get mapped indices of these neighbors in dataset 2
#             mapped_neighbors = [mapped_indices[nbr] for nbr in neighbors1]

#             # Get neighbors of i_prime in dataset 2
#             _, neighbors2 = nbrs2.kneighbors([Y[i_prime]])
#             neighbors2 = neighbors2[0]

#             # Check spatial consistency: how many mapped neighbors are in i_prime's neighborhood
#             spatial_consistency = sum(1 for n_prime in mapped_neighbors if n_prime in neighbors2) / K

#             # Combine label and spatial consistency
#             clc_score += label_consistent * spatial_consistency

#         # Average over all points
#         clc_score /= N
#         results.append({'metric': 'clc', 'value': clc_score, 'group': f'{batch_list[i]}_{batch_list[i+1]}'})
#     df = pd.DataFrame(results)
    
#     return df