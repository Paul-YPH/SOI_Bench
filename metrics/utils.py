import os

os.environ["R_HOME"] = "/usr/lib/R"
os.environ["R_LIBS"] = "/usr/lib/R/library"

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects import NULL

def validate_adata(adata, use_rep=None, batch_key=None, label_key=None, spatial_key=None):
    if not hasattr(adata, "obs"):
        raise ValueError("Input `adata` must be a valid AnnData object.")
    if batch_key is not None and batch_key not in adata.obs.columns:
        raise ValueError(f"Key '{batch_key}' not found in `adata.obs`.")
    if use_rep is not None and use_rep not in adata.obsm.keys():
        raise ValueError(f"Embedding '{use_rep}' not found in `adata.obsm`.")
    if label_key is not None and label_key not in adata.obs.columns:
        raise ValueError(f"Key '{label_key}' not found in `adata.obs`.")
    if spatial_key is not None and spatial_key not in adata.obsm.keys():
        raise ValueError(f"Key '{spatial_key}' not found in `adata.obsm`.")
    
def check_adata(adata, use_rep=None, batch_key=None, label_key=None, spatial_key=None):

    # Initialize a mask for NaN values
    nan_mask = np.zeros(adata.n_obs, dtype=bool)
    nan_details = []

    # Check for NaN in adata.obs[batch_key]
    if batch_key is not None:
        if batch_key in adata.obs:
            nan_mask_obs1 = adata.obs[batch_key].isna()
            nan_mask = nan_mask | nan_mask_obs1
            if nan_mask_obs1.any():
                nan_details.append(
                    f"{nan_mask_obs1.sum()} NaN values found in `adata.obs['{batch_key}']`"
                )
        else:
            print(f"Warning: '{batch_key}' not found in adata.obs.")

    # Check for NaN in adata.obs[label_key]
    if label_key is not None:
        if label_key in adata.obs:
            nan_mask_obs2 = adata.obs[label_key].isna()
            nan_mask = nan_mask | nan_mask_obs2
            if nan_mask_obs2.any():
                nan_details.append(
                    f"{nan_mask_obs2.sum()} NaN values found in `adata.obs['{label_key}']`"
                )
        else:
            print(f"Warning: '{label_key}' not found in adata.obs.")

    # Check for NaN in adata.obsm[use_rep]
    if use_rep is not None:
        if use_rep in adata.obsm:
            nan_mask_obsm = np.isnan(adata.obsm[use_rep]).any(axis=1)
            nan_mask = nan_mask | nan_mask_obsm
            if nan_mask_obsm.any():
                nan_details.append(
                    f"{nan_mask_obsm.sum()} NaN values found in `adata.obsm['{use_rep}']`"
                )
        else:
            print(f"Warning: '{use_rep}' not found in adata.obsm.")

    # Check for NaN in adata.obsm[spatial_key]
    if spatial_key is not None:
        if spatial_key in adata.obsm:
            nan_mask_obsm = np.isnan(adata.obsm[spatial_key]).any(axis=1)
            nan_mask = nan_mask | nan_mask_obsm
            if nan_mask_obsm.any():
                nan_details.append(
                    f"{nan_mask_obsm.sum()} NaN values found in `adata.obsm['{spatial_key}']`"
                )
        else:
            print(f"Warning: '{spatial_key}' not found in adata.obsm.")

    # Filter the AnnData object
    if nan_mask.any():
        print("NaN details:")
        for detail in nan_details:
            print(f" - {detail}")
        print(f"Total {nan_mask.sum()} cells with NaN values found. Removing these cells.")
        adata = adata[~nan_mask].copy()
    else:
        print("No NaN values found.")

    return adata    


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    if res is None or res == NULL:
        print('Mclust fitting failed. We change the modelNames to EEI.')
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, 'EEI')
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=False):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res  
 
def clustering(adata, 
               n_clusters=None, 
               use_rep='integrated', 
               label_key=None, 
               method='mclust', 
               start=0.1, 
               end=3.0, 
               increment=0.01):
    
    print("Starting clustering...")
    validate_adata(adata, use_rep=use_rep, label_key=label_key)
    adata_tmp =check_adata(adata, use_rep=use_rep, label_key=label_key)
    
    if n_clusters is None and label_key is not None:
        n_clusters = adata_tmp.obs[label_key].nunique()
        print(f"Using {n_clusters} clusters from {label_key}")
    else:
        n_clusters = 10
        print("Using default 10 clusters")

    if method == 'mclust':
       print("Running Mclust clustering...")
       adata_tmp = mclust_R(adata_tmp, used_obsm=use_rep, num_cluster=n_clusters)
       print("Mclust clustering completed")
       
    elif method == 'leiden':
       print("Running Leiden clustering...")
       res = search_res(adata_tmp, n_clusters, use_rep=use_rep, method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata_tmp, random_state=0, resolution=res)
       print("Leiden clustering completed")
       
    elif method == 'louvain':
       print("Running Louvain clustering...")
       res = search_res(adata_tmp, n_clusters, use_rep=use_rep, method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata_tmp, random_state=0, resolution=res)
       print("Louvain clustering completed")
       
    # elif method == 'all':
    #     print("Running all clustering methods...")
    #     print("1. Running Mclust...")
    #     adata_tmp = mclust_R(adata_tmp, used_obsm=use_rep, num_cluster=n_clusters)
    #     print("2. Running Leiden...")
    #     res = search_res(adata_tmp, n_clusters, use_rep=use_rep, method='leiden', start=start, end=end, increment=increment)
    #     sc.tl.leiden(adata_tmp, random_state=0, resolution=res)
    #     print("3. Running Louvain...")
    #     res = search_res(adata_tmp, n_clusters, use_rep=use_rep, method='louvain', start=start, end=end, increment=increment)
        
    return adata_tmp

def process_anndata(adata_raw, highly_variable_genes=False, normalize_total=False,
                    log1p=False, scale=False, pca=False, neighbors=False,
                    umap=False, n_top_genes=3000, n_comps=100, ndim=30):

    adata = adata_raw.copy()
    
    if highly_variable_genes:
        if adata.shape[1]>n_top_genes:
            print("Identifying highly variable genes...")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3", span=0.6, min_disp=0.1)
        else:
            adata.var['highly_variable'] = True

    if normalize_total:
        print("Normalizing total counts...")
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)

    if log1p:
        print("Applying log1p transformation...")
        sc.pp.log1p(adata)

    print("Saving pre-log1p counts to a layer...")
    adata.layers["data"] = adata.X.copy()

    if scale:
        print("Scaling the data...")
        sc.pp.scale(adata)

    if pca:
        print("Performing PCA...")
        sc.tl.pca(adata, n_comps=n_comps, svd_solver="auto")
        
    if neighbors:
        print("Calculating neighbors based on cosine metric...")
        sc.pp.neighbors(adata, metric="cosine", n_pcs=ndim)
            
    if umap:
        print("Performing UMAP...")
        sc.tl.umap(adata)
    
    print("Processing completed.")
    return adata

def extract_exp(data, layer=None, gene = None, dataframe = False):
    """
    Extract gene expression data from the given data object.

    Args:
        data (AnnData): AnnData object.
        layer (str): Optional - Layer of data from which to extract expression data. Defaults to None (use data.X).
        gene (str or list): Optional - Gene name or list of gene names to extract expression data for.

    Returns:
        exp_data (pd.DataFrame): DataFrame containing gene expression data.
    """
    if gene is None:
        gene = data.var.index.tolist()
    
    if layer is None:
        if issparse(data.X):
            expression_data = data[:,gene].X.toarray()
        else:
            expression_data = data[:,gene].X
    elif layer in data.layers:
        if issparse(data.layers[layer]):
            expression_data = data[:,gene].layers[layer].toarray()
        else:
            expression_data = data[:,gene].layers[layer]
    else:
        raise ValueError(f"Layer '{layer}' not found in data.layers.")

    if dataframe:
        expression_data = pd.DataFrame(expression_data)
        expression_data.columns = gene
        expression_data.index = data.obs.index.tolist()
    
    return expression_data

def extract_hvg(adata1 ,adata2):

    lst1 = list(adata1.var.index[adata1.var['highly_variable']])
    lst2 = list(adata2.var.index[adata2.var['highly_variable']])
    
    gene_list = list(set(lst1) & set(lst2))
    
    return gene_list

def filter_common_genes(ann_list):
    common_genes = set(ann_list[0].var_names)
    for ann in ann_list[1:]:
        common_genes &= set(ann.var_names)
    common_genes = list(common_genes)
    if not common_genes:
        raise ValueError("No common genes found among the AnnData objects.")
    print(f"Number of common genes: {len(common_genes)}")
    for i in range(len(ann_list)):
        ann_list[i] = ann_list[i][:, common_genes].copy()
    return ann_list

def generate_output_path(base_output_path, angle_true=None, overlap_list=None, pseudocount=None, distortion_list=None,rep_num=None):
    if not os.path.exists(base_output_path):
        raise FileNotFoundError(f"Base output path {base_output_path} does not exist")

    # sample_str = '_'.join(sample_list)
    # sample_path = os.path.join(base_output_path, sample_str)

    sample_path = base_output_path
    
    param_components = []
    if angle_true is not None:
        param_components.append(f"rotation_{'_'.join(map(str, angle_true))}")
    if overlap_list is not None:
        param_components.append(f"overlap_{'_'.join(map(str, overlap_list))}")
    if pseudocount is not None:
        param_components.append(f"pseudocount_{pseudocount}")
    if distortion_list is not None:
        param_components.append(f"distortion_{'_'.join(map(str, distortion_list))}")
    param_str = '_'.join(param_components)
    param_path = os.path.join(sample_path, param_str if param_components else "")
    if rep_num is not None:
        param_path = os.path.join(param_path, f"rep_{rep_num}")
    return param_path

def subsample_adata(ann_list, subsample=False, subsample_num=None):
    if subsample:
        for i in range(len(ann_list)):
            ann = ann_list[i]
            if subsample_num is not None and subsample_num < ann.n_obs:
                random_indices = np.random.choice(ann.n_obs, size=subsample_num, replace=False)
                ann_list[i] = ann[random_indices, :].copy()  
    return ann_list