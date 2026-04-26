import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import gc
import scanpy as sc
from paste2 import PASTE2, projection
from utils import get_ann_list, create_lightweight_adata,set_seed
from paste2.helper import intersect, distance, extract_data_matrix, to_dense_array,generalized_kl_divergence,kl_divergence,high_umi_gene_distance,pca_distance,glmpca_distance
from paste2.PASTE2 import partial_fused_gromov_wasserstein

def partial_pairwise_align(sliceA, sliceB, s, alpha=0.1, armijo=False, dissimilarity='glmpca', use_rep=None, G_init=None, a_distribution=None,b_distribution=None, norm=True, return_obj=False, verbose=True):
    m = s

    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]
    # print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

    # Calculate spatial distances
    D_A = distance.cdist(sliceA.obsm['spatial'], sliceA.obsm['spatial'])
    D_B = distance.cdist(sliceB.obsm['spatial'], sliceB.obsm['spatial'])

    # Calculate expression dissimilarity
    A_X, B_X = to_dense_array(extract_data_matrix(sliceA, use_rep)), to_dense_array(extract_data_matrix(sliceB, use_rep))
    if dissimilarity.lower() == 'euclidean' or dissimilarity.lower() == 'euc':
        M = distance.cdist(A_X, B_X)
    elif dissimilarity.lower() == 'gkl':
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M = generalized_kl_divergence(s_A, s_B)
        M /= M[M > 0].max()
        M *= 10
    elif dissimilarity.lower() == 'kl':
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M = kl_divergence(s_A, s_B)
    elif dissimilarity.lower() == 'selection_kl':
        M = high_umi_gene_distance(A_X, B_X, 2000)
    elif dissimilarity.lower() == "pca":
        M = pca_distance(sliceA, sliceB, 2000, min(min(sliceA.shape)-1, min(sliceB.shape)-1,20))
    elif dissimilarity.lower() == 'glmpca':
        dim = min(min(sliceA.shape)-1, min(sliceB.shape)-1,50)
        data_id = sliceA.obs['data_id'].unique()[0]
        if data_id == 'SD57':
            dim = 10
        M = glmpca_distance(A_X, B_X, latent_dim=dim, filter=True, verbose=verbose)
    else:
        print("ERROR")
        exit(1)

    # init distributions
    if a_distribution is None:
        a = np.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = a_distribution

    if b_distribution is None:
        b = np.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = b_distribution

    if norm:
        D_A /= D_A[D_A > 0].min().min()
        D_B /= D_B[D_B > 0].min().min()

        """
        Code for normalizing distance matrix
        """
        D_A /= D_A[D_A>0].max()
        #D_A *= 10
        D_A *= M.max()
        D_B /= D_B[D_B>0].max()
        #D_B *= 10
        D_B *= M.max()
        """
        Code for normalizing distance matrix ends
        """
    pi, log = partial_fused_gromov_wasserstein(M, D_A, D_B, a, b, alpha=alpha, m=m, G0=G_init, loss_fun='square_loss', armijo=armijo, log=True, verbose=verbose)

    if return_obj:
        return pi, log['partial_fgw_cost']
    return pi


def run_PASTE2(adata, **args_dict):
    seed = args_dict['seed']
    set_seed(seed)
    
    sample_list = adata.uns['config']['sample_list']
    overlap_flag = adata.uns['config']['overlap']
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    pi_list = []
    pi_index_list = []
    pi_column_list = []
    matching_cell_ids_list = []
    
    if overlap_flag:
        overlap_list = []
        for i in range(len(sample_list)-1):
            j = i + 1
            n_i = ann_list[i].shape[0]
            n_j = ann_list[j].shape[0]
            val = min(n_i, n_j) / max(n_i, n_j) 
            overlap_list.append(min(val, 0.99))
    else:
        overlap_list = [0.99] * (len(sample_list) - 1)
        
    for i in range(len(sample_list)-1):
        j = i+1
        adata1 = ann_list[i]
        adata2 = ann_list[j]    

        # l = adata1.copy()
        # siml = adata2.copy()
        #overlap_frac = select_overlap_fraction(l, siml, alpha=0.1)
        sc.pp.filter_genes(adata1, min_cells=3)
        sc.pp.filter_genes(adata2, min_cells=3)
        sc.pp.filter_cells(adata1, min_genes=3)
        sc.pp.filter_cells(adata2, min_genes=3)
        pi12 = partial_pairwise_align(adata1, adata2, dissimilarity = 'pca', s=overlap_list[i])

        cell_ids_1 = adata1.obs.index.to_numpy()
        cell_ids_2 = adata2.obs.index.to_numpy() 

        matching_index = np.argmax(pi12, axis=0)
        matching_cell_ids = pd.DataFrame({
            'id_1': cell_ids_1[matching_index],
            'id_2': cell_ids_2   
        })
        
        pi_list.append(pi12)
        pi_index_list.append(cell_ids_1)
        pi_column_list.append(cell_ids_2)
        matching_cell_ids_list.append(matching_cell_ids)

    new_slices = projection.partial_stack_slices_pairwise(ann_list, pi_list)
    for i in range(0, len(new_slices)):
        new_slices[i].obsm['spatial_aligned'] = new_slices[i].obsm['spatial']
        new_slices[i].obsm['spatial'] = ann_list[i].obsm['spatial']
    adata = sc.concat(new_slices, join='inner', index_unique=None)
    del new_slices
    gc.collect()
    
    adata = create_lightweight_adata(adata, config=config, pi_list=pi_list, pi_index_list=pi_index_list, pi_column_list=pi_column_list, matching_cell_ids_list=matching_cell_ids_list, args_dict=args_dict)
    
    return adata
