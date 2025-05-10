import warnings
warnings.filterwarnings("ignore")

import os
import sys

import pandas as pd
import torch
import scanpy as sc
import harmonypy as hm

import STAGATE_pyG

current_dir = os.path.dirname(os.path.abspath(__file__)) 
benchmark_dir = os.path.dirname(current_dir)  
sys.path.append(benchmark_dir)
from metrics.utils import *

#################### Run STAGATE ####################
def run_STAGATE(ann_list, cluster_option):
    
    for ann in ann_list:
        STAGATE_pyG.Cal_Spatial_Net(ann, k_cutoff=20, model='KNN')
    # Concatenate data
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    adata.uns['Spatial_Net'] = pd.concat([ann.uns['Spatial_Net'] for ann in ann_list])
    # Preprocess data
    print('### Preprocessing data...')
    adata.layers['counts'] = adata.X
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if adata.shape[1]>3000:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    else:
        adata.var['highly_variable'] = True
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
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=cluster_option)
    adata.X = adata.layers['counts']
    return adata