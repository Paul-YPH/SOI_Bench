import scanpy as sc
import torch
import pandas as pd
import numpy as np
import spateo as st
from typing import List
import anndata as ad
from utils import get_ann_list, create_lightweight_adata,set_seed
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# modified from spateo
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
    req_n_comps = int(args.get("n_comps", 50)) 
    max_by_obs = max(1, adata_pca.n_obs - 1)   
    use_hvg_flag = use_hvg                      

    # Identify and use highly variable genes for PCA if requested
    if use_hvg:
        sc.pp.highly_variable_genes(adata_pca, batch_key=batch_key)
        col = hvg_key if hvg_key in adata_pca.var.columns else "highly_variable"
        n_hvg = int(adata_pca.var[col].sum()) if col in adata_pca.var.columns else 0
        if n_hvg < 1:
            use_hvg_flag = False
            n_features = adata_pca.n_vars
        else:
            n_features = n_hvg
    else:
        n_features = adata_pca.n_vars

    n_obs = adata_pca.n_obs
    safe_k_max = min(n_obs, n_features) - 1
    if safe_k_max <= 0:
        print(f"Skipping PCA: matrix shape ({n_obs}, {n_features}) is too small.")
        raise ValueError(f"Cannot run PCA, matrix shape ({n_obs}, {n_features}) is too small.")
    req_n_comps = int(args.get("n_comps", 50))
    n_comps_eff = int(min(req_n_comps, safe_k_max))
    if n_comps_eff <= 0:
        n_comps_eff = safe_k_max 
    args = dict(args)
    args["n_comps"] = n_comps_eff 

    # Perform PCA (use HVG if available & requested)
    sc.tl.pca(adata_pca, **args, use_highly_variable=use_hvg_flag)

    # Split the PCA results back into the original AnnData objects
    for i in range(len(adatas)):
        adatas[i].obsm[pca_key] = adata_pca[adata_pca.obs[batch_key] == str(i)].obsm["X_pca"].copy()
        
def run_Spateo_NR(adata, **args_dict):
    seed = args_dict['seed']
    set_seed(seed)
    pcs = args_dict['pcs']
    knn = args_dict['knn']
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    for ann in ann_list:
        # sc.pp.filter_cells(ann, min_genes=10)
        # sc.pp.filter_genes(ann, min_cells=3)
        sc.pp.normalize_total(ann)
        sc.pp.log1p(ann)
    
    
    min_genes = min([ann.shape[1] for ann in ann_list])
    min_cells = min([ann.shape[0] for ann in ann_list])
    group_pca(ann_list, pca_key='X_pca',n_comps=min(pcs, min_genes-1, min_cells-1),batch_key='batch_tmp')
        
    key_added = 'spatial_aligned'
    # spateo return aligned slices as well as the mapping matrix
    
    total_spots = sum([ann.shape[0] for ann in ann_list])
    is_sparse = False
    if total_spots > 100000:
        is_sparse = True
    
    ann_list, pis = st.align.morpho_align(
        models=ann_list,
        dissimilarity='cos',
        verbose=False,
        spatial_key='spatial',
        key_added=key_added,
        device=device,
        beta=1,
        lambdaVF=1,
        K=knn,
        return_mapping=True,
        rep_layer='X_pca',
        rep_field='obsm',
        sparse_calculation_mode=is_sparse,
        use_chunk=True,
        chunk_capacity=4
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
    
    adata.obsm['spatial_aligned'] = adata.obsm['spatial_aligned_nonrigid']
    
    adata = create_lightweight_adata(adata, config=config, pi_list=pi_list, pi_index_list=pi_index_list, pi_column_list=pi_column_list, matching_cell_ids_list=matching_cell_ids_list, args_dict=args_dict)
    
    return adata