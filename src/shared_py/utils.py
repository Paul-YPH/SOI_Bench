import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy.sparse import issparse
from scipy.sparse import csr_matrix

def get_ann_list(adata):
    config = adata.uns['config']
    data_dict = adata.uns['dataset']
    order_keys = config['sample_list'] 
    ann_list = [data_dict[key] for key in order_keys]
    return ann_list, config

def list_to_dict(data_list):
    if data_list is None:
        return None
    return {str(i): np.array(arr) for i, arr in enumerate(data_list)}

def create_lightweight_adata(adata, config, pi_list=None, pi_index_list=None, pi_column_list=None, matching_cell_ids_list=None, args_dict=None):
    if 'cell_id' in adata.obs.columns:
        if not np.array_equal(adata.obs_names.values, adata.obs['cell_id'].values):
            adata.obs_names = adata.obs['cell_id'].values
            adata.obs.index.name = 'cell_id'
    
    valid_cells = set(adata.obs_names)

    f_pi_list = []
    f_pi_index_list = []
    f_pi_column_list = []
    f_matching_list = []

    if pi_list is not None and pi_index_list is not None and pi_column_list is not None:
        for i in range(len(pi_list)):
            p_mat = pi_list[i]
            p_idx = np.array(pi_index_list[i])
            p_col = np.array(pi_column_list[i])

            row_mask = np.array([c in valid_cells for c in p_idx])
            col_mask = np.array([c in valid_cells for c in p_col])

            p_mat_filtered = p_mat[row_mask, :][:, col_mask]

            f_pi_list.append(p_mat_filtered)
            f_pi_index_list.append(p_idx[row_mask])
            f_pi_column_list.append(p_col[col_mask])

    if matching_cell_ids_list is not None:
        for m_df in matching_cell_ids_list:
            m_df_filtered = m_df[m_df['id_1'].isin(valid_cells) & m_df['id_2'].isin(valid_cells)]
            f_matching_list.append(m_df_filtered)

    var_placeholder = pd.DataFrame(index=adata.var.index)
    adata_meta = anndata.AnnData(
        obs=adata.obs.copy(),
        var=var_placeholder,
        obsm=adata.obsm.copy(),
    )
    
    adata_meta.uns['config'] = config
    
    if pi_list is not None:
        adata_meta.uns['pi_list'] = list_to_dict(f_pi_list)
        adata_meta.uns['pi_index_list'] = list_to_dict(f_pi_index_list)
        adata_meta.uns['pi_column_list'] = list_to_dict(f_pi_column_list)
    
    if matching_cell_ids_list is not None:
        adata_meta.uns['matching_cell_ids_list'] = list_to_dict(f_matching_list)
        
    if args_dict is not None:
        for key, value in args_dict.items():
            adata_meta.uns[key] = value
        filename_parts = [f"{k}_{v}" for k, v in args_dict.items()]
        filename_parts.append('integrated')
        adata_meta.uns['filename'] = '_'.join(filename_parts)
    if 'counts' in adata_meta.layers: del adata_meta.layers['counts']
    return adata_meta

def create_directory(args_dict):
    directory_parts = []
    for key, value in args_dict.items():
        directory_parts.append(f"{key}_{value}")
    directory = '_'.join(directory_parts)
    return directory

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

def process_anndata(adata, highly_variable_genes=False, normalize_total=False,
                    log1p=False, scale=False, pca=False, neighbors=False,
                    umap=False, n_top_genes=3000, n_comps=100, ndim=30):
    
    if highly_variable_genes:
        if adata.shape[1]>n_top_genes:
            print("Identifying highly variable genes...")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3", span=0.6, min_disp=0.1)
            adata = adata[:, adata.var['highly_variable']].copy()
        else:
            adata.var['highly_variable'] = True

    if normalize_total:
        print("Normalizing total counts...")
        sc.pp.normalize_total(adata)

    if log1p:
        print("Applying log1p transformation...")
        sc.pp.log1p(adata)

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


import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed) 
        print(f"[Seed Util] PyTorch seed set to {seed}")
    except ImportError:
        pass

import anndata as ad
def make_adata_with_config(adata_dict,
                adata_path = None,
                distortion = False,
                multi_slice = False,
                multiomics_cross_slice = False,
                multiomics_one_slice = False,
                overlap = False,
                pseudocount = False,
                rotation = False,
                sample_list = None):
    adata = ad.AnnData()
    adata.uns = {}
    adata.uns['config'] = {}
    adata.uns['dataset'] = adata_dict
    adata.uns['config']['adata_path'] = adata_path
    adata.uns['config']['distortion'] = distortion
    adata.uns['config']['multi_slice'] = multi_slice
    adata.uns['config']['multiomics_cross_slice'] = multiomics_cross_slice
    adata.uns['config']['multiomics_one_slice'] = multiomics_one_slice
    adata.uns['config']['overlap'] = overlap
    adata.uns['config']['pseudocount'] = pseudocount
    adata.uns['config']['rotation'] = rotation
    adata.uns['config']['sample_list'] = sample_list
    
    return adata

def add_obs(adata, 
            batch = None,
            technology = None, 
            species = None, 
            ground_truth = None,
            angle = None,
            overlap = None,
            distortion = None,
            pseudocount = None,
            data_id = None
            ):
    # change obs
    for col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype(str)
    for col in adata.var.columns:
        adata.var[col] = adata.var[col].astype(str)
        
    # change obsm
    for key in adata.obsm.keys():
        adata.obsm[key] = adata.obsm[key].astype(np.float32)
        
    if batch is not None:
        adata.obs['batch'] = batch
    if technology is not None:
        adata.obs['technology'] = technology
    if species is not None:
        adata.obs['species'] = species
    if ground_truth is not None:
        adata.obs['Ground Truth'] = ground_truth
    if angle is not None:
        adata.obs['angle'] = angle
    if overlap is not None:
        adata.obs['overlap'] = overlap
    if distortion is not None:
        adata.obs['distortion'] = distortion
    if pseudocount is not None:
        adata.obs['pseudocount'] = pseudocount
    if data_id is not None:
        adata.obs['data_id'] = data_id
    
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    adata.X = adata.X.astype(np.float32)
    
    if 'array_col' not in adata.obs.columns:
        adata.obs['array_col'] = adata.obsm['spatial'][:, 1]
    else:
        adata.obs['array_col'] = adata.obs['array_col'].astype(np.float32)
    if 'array_row' not in adata.obs.columns:
        adata.obs['array_row'] = adata.obsm['spatial'][:, 0] 
    else:
        adata.obs['array_row'] = adata.obs['array_row'].astype(np.float32)
        
        
    # change obs_names
    first_batch = str(adata.obs['batch'][0])
    first_name = str(adata.obs_names[0])
    if not first_name.startswith(first_batch+'_'):
        adata.obs_names = adata.obs['batch'].astype(str) + '_' + adata.obs_names
    adata.obs['cell_id'] = adata.obs_names

    return adata


import os, time, resource, threading
import pynvml

class ResourceMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.start_time = time.time()
        self.pid = os.getpid()
        self.max_gpu_memory = 0
        self.gpu_model = "Unknown"  
        self.running = False
        self.thread = None

    def _monitor_thread(self):
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
            
            if device_count > 0:
                model_name = pynvml.nvmlDeviceGetName(handles[0])
                self.gpu_model = model_name.decode('utf-8') if isinstance(model_name, bytes) else model_name
            # -----------------------------------------------------------

            max_gpu = 0
            while self.running:
                try:
                    current_max = 0
                    for h in handles:
                        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                        for p in procs:
                            if p.pid == self.pid:
                                m = p.usedGpuMemory / (1024 * 1024) 
                                current_max += m
                    max_gpu = max(max_gpu, current_max)
                except pynvml.NVMLError:
                    pass
                time.sleep(self.interval)
            
            self.max_gpu_memory = round(max_gpu, 2)
        except Exception as e:
            print(f"GPU Monitor Error: {e}")
        finally:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_thread, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        self.end_time = time.time()
        try:
            usage_self = resource.getrusage(resource.RUSAGE_SELF)
            usage_children = resource.getrusage(resource.RUSAGE_CHILDREN)
            self.max_cpu_mb = round((usage_self.ru_maxrss + usage_children.ru_maxrss) / 1024, 2)
            self.duration_min = round((self.end_time - self.start_time) / 60, 2)
        except:
            self.max_cpu_mb = 0
            self.duration_min = 0