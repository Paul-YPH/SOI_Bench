import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import scanpy as sc
import numpy as np
import gc
from SpatialGlue.preprocess import clr_normalize_each_cell, pca, lsi
from SpatialGlue.preprocess import construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
from clustering import clustering
from utils import get_ann_list, create_lightweight_adata,set_seed

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def run_SpatialGlue(adata, **args_dict):
    seed = args_dict['seed']
    clust = args_dict['clust']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    # ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome','SPOTS']
    
    tech = ann_list[0].obs['technology'].unique()[0]
    if tech in ['spatialatacrnaseq', 'misarseq']:
        data_type = 'Spatial-epigenome-transcriptome'
        knn=6
    else:
        data_type = 'SPOTS'
        knn=3
        
    adata_omics1 = ann_list[config['sample_list'].tolist().index('rna')]
    if data_type == 'Spatial-epigenome-transcriptome':
        adata_omics2 = ann_list[config['sample_list'].tolist().index('atac')]
    else:
        adata_omics2 = ann_list[config['sample_list'].tolist().index('adt')]

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    adata_omics1.obs_names = [n.split('_', 1)[1] for n in adata_omics1.obs_names]
    adata_omics2.obs_names = [n.split('_', 1)[1] for n in adata_omics2.obs_names]

    sc.pp.filter_genes(adata_omics1, min_cells=1)
    sc.pp.filter_cells(adata_omics1, min_genes=1)

    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=min(3000, adata_omics1.shape[1], adata_omics1.shape[0]))
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
    adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)

    common_index = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
    if len(common_index) == 0:
        def strip_prefix(names):
            return [n.split('-', 1)[1] if '-' in n else n for n in names]
        bc1 = strip_prefix(adata_omics1.obs_names)
        bc2 = strip_prefix(adata_omics2.obs_names)
        common_bc = set(bc1) & set(bc2)
        idx1 = [i for i, b in enumerate(bc1) if b in common_bc]
        bc2_map = {b: i for i, b in enumerate(bc2)}
        idx2 = [bc2_map[bc1[i]] for i in idx1]
        adata_omics1 = adata_omics1[adata_omics1.obs_names[idx1]].copy()
        adata_omics2 = adata_omics2[adata_omics2.obs_names[idx2]].copy()
    else:
        adata_omics1 = adata_omics1[common_index].copy()
        adata_omics2 = adata_omics2[common_index].copy()

    del ann_list
    gc.collect()
    
    # ATAC
    if data_type == 'Spatial-epigenome-transcriptome':
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=min(3000, adata_omics2.shape[1], adata_omics2.shape[0]))
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
    elif data_type == 'SPOTS':
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)
    
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type, n_neighbors=knn)
    
    # define model
    model = Train_SpatialGlue(data, datatype=data_type, device=device)

    # train model
    output = model.train()
    
    adata = adata_omics1.copy()
    del adata_omics1, adata_omics2
    
    adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1']
    adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2']
    adata.obsm['SpatialGlue'] = output['SpatialGlue']
    adata.obsm['alpha'] = output['alpha']
    adata.obsm['alpha_omics1'] = output['alpha_omics1']
    adata.obsm['alpha_omics2'] = output['alpha_omics2']

    n_clusters = int(len(adata.obs['Ground Truth'].unique()))

    adata.obsm[ 'SpatialGlue_pca'] = pca(adata, use_reps="SpatialGlue", n_comps=20)
    adata = clustering(adata, use_rep='SpatialGlue_pca', label_key='Ground Truth', method=clust, mclust_version = "2")
    
    adata.obsm['integrated'] = adata.obsm['SpatialGlue']
    adata.obs['benchmark_cluster'] = adata.obs[clust]

    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    return adata
