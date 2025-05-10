import os
import scanpy as sc
import torch
import pandas as pd
import numpy as np

import spateo as st

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#################### Run Spateo ####################
def run_Spateo_R(ann_list):
    for ann in ann_list:
        # sc.pp.filter_cells(ann, min_genes=10)
        # sc.pp.filter_genes(ann, min_cells=3)
        ann.layers["counts"] = ann.X.copy()
        sc.pp.normalize_total(ann)
        sc.pp.log1p(ann)
        if ann.shape[1]>3000:
            sc.pp.highly_variable_genes(ann, n_top_genes=3000)
        else:
            ann.var['highly_variable'] = True

    # st.align.group_pca(ann_list, pca_key='X_pca')
    key_added = 'spatial_aligned'
    # spateo return aligned slices as well as the mapping matrix
    ann_list, pis = st.align.morpho_align(
        models=ann_list,
        ## Uncomment this if use highly variable genes
        # models=[slice1[:, slice1.var.highly_variable], slice2[:, slice2.var.highly_variable]],
        ## Uncomment the following if use pca embeddings
        # rep_layer='X_pca',
        # rep_field='obsm',
        # dissimilarity='cos',
        sparse_calculation_mode=True,
        use_chunk=True,
        chunk_capacity=4,
        verbose=False,
        spatial_key='spatial',
        key_added=key_added,
        device=device,
        return_mapping=True
    )
    for ann in ann_list:
        ann.X = ann.layers["counts"]
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    
    pi_list = []
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
            'cell_id_1': cell_ids_1[matching_index],
            'cell_id_2': cell_ids_2   
        })

        pi_list.append(result)
        matching_cell_ids_list.append(matching_cell_ids)
    return adata, pi_list, matching_cell_ids_list