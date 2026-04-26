import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import warnings
import numpy as np
from scipy.sparse import block_diag
from sklearn.decomposition import PCA
import scipy.sparse as sp
import spCLUE
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering   

spCLUE.fix_seed(0)

def run_spCLUE(adata, **args_dict):
    seed = args_dict['seed']
    knn = args_dict['knn']
    pcs = args_dict['pcs']
    clust = args_dict['clust']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    batch_list = []
    for i, ann in enumerate(ann_list):
        print(f"|================== Current slice index: {i} ===============|")
        ann.var_names_make_unique()
        ann.layers['count'] = ann.X.copy()
        ann_list[i] = spCLUE.preprocess(ann,hvgNumber=min(3000, ann.shape[1],ann.shape[0]))
        ann_list[i].obs["batch"] = str(i)  
        batch_list += [i] * ann.shape[0]

    batch_list = np.array(batch_list)

    # spatial graph
    g_spatial_list = []
    for adata_cur in ann_list:
        g_spatial = spCLUE.prepare_graph(adata_cur, "spatial",n_neighbors=knn)
        g_spatial_list.append(g_spatial)
    g_spatial = block_diag(g_spatial_list)

    # expression graph
    g_expr_list = []
    for adata_cur in ann_list:
        g_expr = spCLUE.prepare_graph(adata_cur, "expr",n_neighbors=knn,n_comps = min(pcs, adata_cur.shape[1]-1, adata_cur.shape[0]-1))
        g_expr_list.append(g_expr)
    g_expr = block_diag(g_expr_list)
    graph_dict = {"spatial": g_spatial, "expr": g_expr}

    adata = sc.concat(ann_list)
    adata.obsm["X_pca"] = PCA(n_components=min(pcs, adata.shape[1]-1, adata.shape[0]-1)).fit_transform(adata.X)
    del ann_list
    gc.collect()
    
    n_clusters = adata.obs["Ground Truth"].nunique()
    spCLUE_model = spCLUE.spCLUE(
        adata.obsm["X_pca"],
        dim_input=adata.obsm["X_pca"].shape[1],
        graph_dict=graph_dict,
        n_clusters=n_clusters,
        batch_list=batch_list,
        batch_train=True
    )
    _, adata.obsm["spCLUE"] = spCLUE_model.trainBatch()
    adata.obs["batch_name"] = batch_list
    adata.obs["batch_name"] = adata.obs["batch_name"].astype("category")
    
    adata.obsm["integrated"] = adata.obsm["spCLUE"]

    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)
    spCLUE.batch_refine_label(adata, key=clust, batch_key="batch_name")
    adata.obs['benchmark_cluster'] = adata.obs[clust+'_refined']
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata