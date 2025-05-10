import os
import scanpy as sc
import torch
import pandas as pd
import numpy as np

import spateo as st
from typing import List
import anndata as ad

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#################### Run Spateo ####################
def group_pca(
    adatas: List[ad.AnnData],
    batch_key: str = "batch",
    pca_key: str = "X_pca",
    use_hvg: bool = True,
    hvg_key: str = "highly_variable",
    **args,
) -> None:
    # Check if batch_key already exists in any of the adatas
    for i, adata in enumerate(adatas):
        if batch_key in adata.obs.columns:
            raise ValueError(
                f"batch_key '{batch_key}' already exists in adata.obs for dataset {i}. Please choose a different key."
            )
    # Concatenate all AnnData objects, using batch_key to differentiate them
    adata_pca = ad.concat(adatas, label=batch_key)
    
    if adata_pca.shape[1]>3000:   
        sc.pp.highly_variable_genes(adata_pca, n_top_genes=3000)
    else:
        adata_pca.var['highly_variable'] = True
        
    # Identify and use highly variable genes for PCA if requested
    if use_hvg:
        sc.tl.pca(adata_pca, **args, use_highly_variable=True)
    else:
        # Perform PCA without restricting to highly variable genes
        sc.tl.pca(adata_pca, **args)
    # Split the PCA results back into the original AnnData objects
    for i in range(len(adatas)):
        adatas[i].obsm[pca_key] = adata_pca[adata_pca.obs[batch_key] == str(i)].obsm["X_pca"].copy()
        
def run_Spateo_NR(ann_list):
    for ann in ann_list:
        # sc.pp.filter_cells(ann, min_genes=10)
        # sc.pp.filter_genes(ann, min_cells=3)
        ann.layers["counts"] = ann.X.copy()
        sc.pp.normalize_total(ann)
        sc.pp.log1p(ann)
        
    group_pca(ann_list, pca_key='X_pca',n_comps=20,batch_key='batch_tmp')
        
    key_added = 'spatial_aligned'
    # spateo return aligned slices as well as the mapping matrix
    ann_list, pis = st.align.morpho_align(
        models=ann_list,
        dissimilarity='cos',
        verbose=False,
        spatial_key='spatial',
        key_added=key_added,
        device=device,
        beta=1,
        lambdaVF=1,
        K=15,
        return_mapping=True,
        rep_layer='X_pca',
        rep_field='obsm',
        sparse_calculation_mode=True,
        use_chunk=True,
        chunk_capacity=4
    )
    for ann in ann_list:
        ann.X = ann.layers["counts"]
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    adata.obsm['spatial_aligned'] = adata.obsm['spatial_aligned_nonrigid']
    
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