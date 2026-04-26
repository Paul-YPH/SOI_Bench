import scanpy as sc
import torch
import pandas as pd
import numpy as np
import spateo as st
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def run_Spateo_R(adata, **args_dict):
    seed = args_dict['seed']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    for ann in ann_list:
        # sc.pp.filter_cells(ann, min_genes=10)
        # sc.pp.filter_genes(ann, min_cells=3)
        sc.pp.normalize_total(ann)
        sc.pp.log1p(ann)
        sc.pp.highly_variable_genes(ann, n_top_genes=min(3000, ann.shape[1], ann.shape[0]))

    # st.align.group_pca(ann_list, pca_key='X_pca')
    key_added = 'spatial_aligned'
    # spateo return aligned slices as well as the mapping matrix
    
    total_spots = sum([ann.shape[0] for ann in ann_list])
    is_sparse = False
    if total_spots > 100000:
        is_sparse = True
            
    ann_list, pis = st.align.morpho_align(
        models=ann_list,
        ## Uncomment this if use highly variable genes
        # models=[slice1[:, slice1.var.highly_variable], slice2[:, slice2.var.highly_variable]],
        ## Uncomment the following if use pca embeddings
        # rep_layer='X_pca',
        # rep_field='obsm',
        # dissimilarity='cos',
        sparse_calculation_mode=is_sparse,
        use_chunk=True,
        chunk_capacity=4,
        verbose=False,
        spatial_key='spatial',
        key_added=key_added,
        device=device,
        return_mapping=True
    )
    
    pi_list = []
    pi_index_list = []
    pi_column_list = []
    matching_cell_ids_list = []
    
    import scipy.sparse as sp
    for i in range(len(pis)):
        if sp.issparse(pis[i]):  
            pis[i] = pis[i].tocoo()  
            result = pd.DataFrame(pis[i].toarray()) 
        else:
            result = pd.DataFrame(pis[i])  
        cell_ids_1 = ann_list[i].obs.index.to_numpy()
        cell_ids_2 = ann_list[i+1].obs.index.to_numpy() 

        result.index = cell_ids_1
        result.columns = cell_ids_2

        matching_index = np.argmax(result.to_numpy(), axis=0)
        matching_cell_ids = pd.DataFrame({
            'id_1': cell_ids_1[matching_index],
            'id_2': cell_ids_2   
        })

        pi_list.append(result.to_numpy())
        pi_index_list.append(cell_ids_1)
        pi_column_list.append(cell_ids_2)
        matching_cell_ids_list.append(matching_cell_ids)
        
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    del ann_list
    gc.collect()
    
    adata = create_lightweight_adata(adata, config=config, pi_list=pi_list, pi_index_list=pi_index_list, pi_column_list=pi_column_list, matching_cell_ids_list=matching_cell_ids_list, args_dict=args_dict)
    
    return adata