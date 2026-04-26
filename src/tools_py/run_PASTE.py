import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import ot
import paste as pst
import scipy
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed

def run_PASTE(adata, **args_dict):
    seed = args_dict['seed']
    set_seed(seed)
    
    sample_list = adata.uns['config']['sample_list']
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    pi_list = []
    pi_list_index = []
    pi_list_columns = []
    matching_cell_ids_list = []
    for i in range(len(sample_list)-1):
        j = i+1
        adata1 = ann_list[i]
        adata2 = ann_list[j]
        
        # def match_spots_using_spatial_heuristic(
        #     X,
        #     Y,
        #     use_ot: bool = True) -> np.ndarray:
        #     n1,n2=len(X),len(Y)
        #     X,Y = norm_and_center_coordinates(X),norm_and_center_coordinates(Y)
        #     dist = scipy.spatial.distance_matrix(X,Y)
        #     if use_ot:
        #         pi = ot.emd(np.ones(n1)/n1, np.ones(n2)/n2, dist)
        #     else:
        #         row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(scipy.sparse.csr_matrix(dist))
        #         pi = np.zeros((n1,n2))
        #         pi[row_ind, col_ind] = 1/max(n1,n2)
        #         if n1<n2: pi[:, [(j not in col_ind) for j in range(n2)]] = 1/(n1*n2)
        #         elif n2<n1: pi[[(i not in row_ind) for i in range(n1)], :] = 1/(n1*n2)
        #     return pi
        # def norm_and_center_coordinates(X):
        #     if min(scipy.spatial.distance.pdist(X))==0:
        #         return X
        #     else:
        #         return (X-X.mean(axis=0))/min(scipy.spatial.distance.pdist(X))
        
        # pi0 = match_spots_using_spatial_heuristic(adata1.obsm['spatial'],adata2.obsm['spatial'],use_ot=True)
        # pi12 = pst.pairwise_align(adata1, adata2, G_init=pi0, norm=True, backend=ot.backend.TorchBackend(), use_gpu = True)

        # pi12 = pst.pairwise_align(adata1, adata2, norm=True, backend=ot.backend.TorchBackend(), use_gpu = True)
        pi12 = pst.pairwise_align(adata1, adata2, norm=True, backend=ot.backend.TorchBackend(), use_gpu = True,dissimilarity = "Euclidean")

        cell_ids_1 = adata1.obs.index.to_numpy()
        cell_ids_2 = adata2.obs.index.to_numpy() 

        matching_index = np.argmax(pi12, axis=0)
        matching_cell_ids = pd.DataFrame({
            'id_1': cell_ids_1[matching_index],
            'id_2': cell_ids_2   
        })
        
        pi_list.append(pi12)
        pi_list_index.append(cell_ids_1)
        pi_list_columns.append(cell_ids_2)
        matching_cell_ids_list.append(matching_cell_ids)
            
    new_slices = pst.stack_slices_pairwise(ann_list, pi_list)
    for i in range(0, len(new_slices)):
        new_slices[i].obsm['spatial_aligned'] = new_slices[i].obsm['spatial']
        new_slices[i].obsm['spatial'] = ann_list[i].obsm['spatial']
    adata = sc.concat(new_slices, join='inner', index_unique=None)
    del new_slices
    gc.collect()
    
    adata = create_lightweight_adata(adata, config=config, pi_list=pi_list, pi_index_list=pi_list_index, pi_column_list=pi_list_columns, matching_cell_ids_list=matching_cell_ids_list, args_dict=args_dict)
    
    return adata