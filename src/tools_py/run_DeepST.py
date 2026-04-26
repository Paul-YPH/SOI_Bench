import warnings
warnings.filterwarnings("ignore")

import types
import numpy as np
import scanpy as sc
import deepstkit as dt
import gc
from scipy.sparse import issparse
from utils import get_ann_list, create_lightweight_adata,filter_common_genes,set_seed

def _leiden(adata, res):
    sc.tl.leiden(adata, random_state=0, resolution=res,
                 flavor="igraph", n_iterations=2, directed=False)
    return len(adata.obs["leiden"].unique())


def _priori_cluster_fixed(self, adata, n_domains):
    """
    Binary search over resolution to find the value that yields exactly
    n_domains Leiden clusters. Resolution range: [0.001, 2.5].
    Leiden cluster count is monotonically non-decreasing with resolution.
    """
    lo, hi = 0.001, 2.5
    tol = 1e-3

    # Quick boundary check
    if _leiden(adata, lo) > n_domains:
        print(f"Warning: even resolution={lo} gives more than {n_domains} domains")
        return lo
    if _leiden(adata, hi) < n_domains:
        print(f"Warning: even resolution={hi} gives fewer than {n_domains} domains")
        return hi

    best_res = lo
    while hi - lo > tol:
        mid = (lo + hi) / 2
        n = _leiden(adata, mid)
        if n == n_domains:
            best_res = mid
            break
        elif n < n_domains:
            lo = mid
        else:
            hi = mid
    else:
        # Tolerance reached without exact match; try rounding to nearest 0.01
        for res in np.arange(round(lo, 2), round(hi, 2) + 0.01, 0.01):
            if _leiden(adata, res) == n_domains:
                best_res = res
                break

    print(f"Found resolution: {best_res:.3f} for {n_domains} domains")
    return best_res

def run_DeepST(adata, output_path, **args_dict):
    
    seed = args_dict['seed']
    knn = args_dict['knn']
    pcs = args_dict['pcs']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    ann_list = filter_common_genes(ann_list)
    del adata
    gc.collect()

    for ann in ann_list:    
        if issparse(ann.X):
            ann.X = ann.X.toarray()

    deepen = dt.main.run(save_path = output_path,
	task = "Integration",
	pre_epochs = 800,
	epochs = 1000,
	use_gpu = True,
	)
    deepen._priori_cluster = types.MethodType(_priori_cluster_fixed, deepen)
    graph_list = []
    sample_list = []
    for i, ann in enumerate(ann_list):
        # ann = process_h5ad(ann, quality="hires")

        ann_list[i] = deepen._get_augment(ann,spatial_type="KDTree",use_morphological=False)
        graph_dict = deepen._get_graph(ann_list[i].obsm['spatial'], distType='KDTree', k= knn) ### k
        # ann = deepen._get_image_crop(ann, data_name=ann.obs["batch"].values[0])
        # ann = deepen._get_augment(ann, spatial_type="LinearRegress")
        # graph_dict = deepen._get_graph(ann.obsm["spatial"], distType = "KDTree")
        graph_list.append(graph_dict) 
        sample_list.append(ann_list[i].obs["batch"].values[0])

    adata, multiple_graph = deepen._get_multiple_adata(adata_list = ann_list, data_name_list = sample_list, graph_list = graph_list)
    del ann_list
    gc.collect()
    
    data = deepen._data_process(adata, pca_n_comps = min(pcs, adata.shape[1]-1, adata.shape[0]-1)) ### pca
    # data = np.ascontiguousarray(data)
    
    deepst_embed = deepen._fit(
            data = data,
            graph_dict = multiple_graph,
            domains = adata.obs["batch"].values,  
            n_domains = len(sample_list))
    adata.obsm["DeepST_embed"] = deepst_embed
    adata.obsm["integrated"] = adata.obsm["DeepST_embed"]
    # Clustering
    # adata.obs_names = adata.obs['cell_id']
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    adata = deepen._get_cluster_data(adata, n_domains=adata.obs["Ground Truth"].nunique(), priori = True)
    adata.obs['benchmark_cluster'] = adata.obs['DeepST_refine_domain']
    
    return adata