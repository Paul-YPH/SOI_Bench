import warnings
warnings.filterwarnings("ignore")

import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA 
import harmonypy as hm
import scipy

import SEDR

current_dir = os.path.dirname(os.path.abspath(__file__)) 
benchmark_dir = os.path.dirname(current_dir)  
sys.path.append(benchmark_dir)
from metrics.utils import *

#################### Run SDER ####################
def run_SDER(ann_list, sample_list, cluster_option):
    for ann in ann_list:
        graph_dict_tmp = SEDR.graph_construction(ann, 12)

        sample = ann.obs['batch'].unique()[0]
        if sample == sample_list[0]:
            adata = ann
            graph_dict = graph_dict_tmp
            name = sample
            adata.obs['proj_name'] = sample
            
        else:
            var_names = adata.var_names.intersection(ann.var_names)
            adata = adata[:, var_names]
            ann = ann[:, var_names]
            ann.obs['proj_name'] = sample

            adata = adata.concatenate(ann, batch_categories=None)
            graph_dict = SEDR.combine_graph_dict(graph_dict, graph_dict_tmp)
            name = name + '_' + sample
    # Preprocess data
    adata.layers['count'] = adata.X.toarray() if isinstance(adata.X, scipy.sparse.spmatrix) else adata.X
    # sc.pp.filter_genes(adata, min_cells=50)
    # sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    if adata.shape[1]>3000:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    else:
        adata.var['highly_variable'] = True
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)
    # Perform PCA
    adata_X = PCA(n_components=min(200, adata.shape[1]), random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X
    # Perform SEDR
    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device='cuda:0')
    using_dec = False
    if using_dec:
        sedr_net.train_with_dec()
    else:
        sedr_net.train_without_dec()
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['SEDR'] = sedr_feat

    meta_data = adata.obs[['batch']]
    data_mat = adata.obsm['SEDR']
    vars_use = ['batch']
    ho = hm.run_harmony(data_mat, meta_data, vars_use)
    res = pd.DataFrame(ho.Z_corr).T
    res_df = pd.DataFrame(data=res.values, columns=['X{}'.format(i+1) for i in range(res.shape[1])], index=adata.obs.index)
    adata.obsm['integrated'] = res_df.values
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=cluster_option)
    adata.X = adata.layers['count']
    return adata