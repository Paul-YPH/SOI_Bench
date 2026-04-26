import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import scanpy as sc
import harmonypy as hm
import STAGATE_pyG
import gc
from utils import set_seed, get_ann_list, create_lightweight_adata
from clustering import clustering  

def run_STAGATE(adata, **args_dict):
    clust = args_dict['clust']
    knn = args_dict['knn']
    seed = args_dict['seed']
    set_seed(seed)
    
    print(f"Running STAGATE with cluster_option: {clust}, knn: {knn}, seed: {seed}")
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    for ann in ann_list:
        STAGATE_pyG.Cal_Spatial_Net(ann, k_cutoff=knn,model='KNN')
    # Concatenate data
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    adata.uns['Spatial_Net'] = pd.concat([ann.uns['Spatial_Net'] for ann in ann_list])
    
    del ann_list
    gc.collect()
    
    # Preprocess data
    print('### Preprocessing data...')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=min(3000, adata.shape[1]))
    # Train STAGATE
    print('### Performing STAGATE...')
    adata = STAGATE_pyG.train_STAGATE(adata, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    ### filter the anndata according to the stagate embedding
    embedding = adata.obsm['STAGATE']
    row_norms = np.linalg.norm(embedding, axis=1)
    non_zero_mask = row_norms != 0
    adata = adata[non_zero_mask].copy()
    
    meta_data = adata.obs[['batch']]
    data_mat = adata.obsm['STAGATE']
    vars_use = ['batch']
    
    ho = hm.run_harmony(data_mat, meta_data, vars_use)
    res = pd.DataFrame(ho.Z_corr).T
    res_df = pd.DataFrame(data=res.values, columns=['X{}'.format(i+1) for i in range(res.shape[1])], index=adata.obs.index)
    adata.obsm['integrated'] = res_df.values
    
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    
    adata = create_lightweight_adata(adata,config,args_dict=args_dict)
    
    return adata