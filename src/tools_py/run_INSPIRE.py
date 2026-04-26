import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse
import INSPIRE
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering   
from sklearn.metrics import pairwise_distances

# https://inspire-tutorial.readthedocs.io/en/latest/tutorials/3d_reconstruction/3d_stereoseq_rigid.html

def prepare_inputs_LGCN(adata_list, num_hvgs, rad_coef=1.1, min_genes_qc=1, min_cells_qc=1, min_concat_dist=50):
    n_slices = len(adata_list)
    adata_list = [adata.copy() for adata in adata_list]
    
    loc_ref = np.array(adata_list[0].obsm["spatial"]).copy()
    pair_dist_ref = pairwise_distances(loc_ref)
    min_dist_ref = np.sort(np.unique(pair_dist_ref), axis=None)[1]
    rad_cutoff = min_dist_ref * rad_coef
    
    for i in range(n_slices):
        adata = adata_list[i]
        adata.var_names_make_unique()
        adata = adata[:, np.array(~adata.var.index.isna())
                       & np.array(~adata.var_names.str.startswith("mt-"))
                       & np.array(~adata.var_names.str.startswith("MT-"))]
        
        sc.pp.filter_cells(adata, min_genes=min_genes_qc)
        sc.pp.filter_genes(adata, min_cells=min_cells_qc)
        
        if scipy.sparse.issparse(adata.X):
            adata.layers["raw_counts"] = adata.X.copy()
        else:
            adata.layers["raw_counts"] = adata.X.copy()
        
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray()
        
        loc = adata.obsm["spatial"]
        pair_dist = pairwise_distances(loc)
        G = (pair_dist < rad_cutoff).astype(float)
        del pair_dist
        
        deg = np.sum(G, axis=1)
        sG = scipy.sparse.coo_matrix(G)
        n_spots = G.shape[0]
        del G
        row, col, edge_weight = sG.row, sG.col, sG.data
        del sG
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        adjG = scipy.sparse.coo_array((edge_weight, (row, col)), shape=(n_spots, n_spots)).toarray()
        
        adata.obsm["node_features_AX"] = adjG @ adata.X
        adata_list[i] = adata

    for i in range(n_slices):
        adata = adata_list[i]
        adata_temp = ad.AnnData(X=adata.layers["raw_counts"].copy())
        adata_temp.var_names = adata.var_names
        adata_temp.obs_names = adata.obs_names
        sc.pp.highly_variable_genes(adata_temp, flavor='seurat_v3', n_top_genes=num_hvgs)
        hvgs = adata_temp.var[adata_temp.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        if i == 0:
            hvgs_shared = set(hvgs)
        else:
            hvgs_shared = hvgs_shared & set(hvgs)
    hvgs_shared = sorted(list(hvgs_shared))

    adata_st_list = []
    for i in range(n_slices):
        adata = adata_list[i]
        st_mtx = adata[:, hvgs_shared].layers["raw_counts"].copy()
        if scipy.sparse.issparse(st_mtx):
            st_mtx = st_mtx.toarray()
        
        adata_st_slice = ad.AnnData(np.zeros(st_mtx.shape))
        adata_st_slice.var.index = hvgs_shared
        adata_st_slice.obs.index = adata.obs.index
        adata_st_slice.obsm["count"] = st_mtx
        adata_st_slice.obs["library_size"] = np.sum(st_mtx, axis=1)
        
        adata_temp = ad.AnnData(X=st_mtx)
        sc.pp.normalize_total(adata_temp, target_sum=1e4)
        sc.pp.log1p(adata_temp)
        st_lognorm = adata_temp.X.copy()
        if scipy.sparse.issparse(st_lognorm):
            st_lognorm = st_lognorm.toarray()
        adata_st_slice.obsm["node_features"] = st_lognorm
        adata_st_slice.obsm["spatial"] = adata.obsm["spatial"].copy()
        adata_st_slice.obs['slice'] = i
        adata_st_slice.obs["slice"] = adata_st_slice.obs["slice"].values.astype(int)
        adata_st_list.append(adata_st_slice)

    for i in range(n_slices):
        hvg_idx = [adata_list[i].var_names.tolist().index(g) for g in hvgs_shared]
        node_fts = adata_list[i].obsm["node_features_AX"][:, hvg_idx].copy()
        if scipy.sparse.issparse(node_fts):
            node_fts = node_fts.toarray()
        adata_st_list[i].obsm["node_features"] = np.concatenate([adata_st_list[i].obsm["node_features"], node_fts], axis=1)
        adata_st_list[i].obs.index = adata_st_list[i].obs.index + "-" + str(i)

    for i in range(n_slices):
        if i > 0:
            xmax_1 = np.max(adata_st_list[i-1].obsm["spatial"][:,0])
            xmin_2 = np.min(adata_st_list[i].obsm["spatial"][:,0])
            ymax_1 = np.max(adata_st_list[i-1].obsm["spatial"][:,1])
            ymax_2 = np.max(adata_st_list[i].obsm["spatial"][:,1])
            adata_st_list[i].obsm["spatial"][:,0] = adata_st_list[i].obsm["spatial"][:,0] + (xmax_1 - xmin_2) + min_concat_dist
            adata_st_list[i].obsm["spatial"][:,1] = adata_st_list[i].obsm["spatial"][:,1] + (ymax_1 - ymax_2)
    
    loc_full = np.concatenate([adata_st_list[i].obsm["spatial"] for i in range(n_slices)], axis=0)
    for i in range(n_slices):
        ad_tmp = ad.AnnData(np.zeros((adata_st_list[i].shape[0], 1)))
        ad_tmp.obs.index = adata_st_list[i].obs.index
        ad_tmp.var.index = ["gene1"]
        ad_tmp.obs["slice_label"] = i
        ad_tmp.obs["slice_label"] = ad_tmp.obs["slice_label"].values.astype(int)
        if i == 0:
            adata_viz = ad_tmp.copy()
        else:
            adata_viz = ad.concat([adata_viz, ad_tmp.copy()], join="outer")
    del ad_tmp
    adata_viz.obsm["spatial"] = loc_full

    return adata_st_list, adata_viz

def spatial_registration(adata_full, batch_key='batch'):
    adata_light = ad.AnnData(obs=adata_full.obs.copy(), obsm=adata_full.obsm.copy())
    
    if "spatial_regi" not in adata_light.obsm:
        adata_light.obsm["spatial_regi"] = adata_light.obsm["spatial"].copy()

    slices = list(set(adata_light.obs[batch_key]))
    slices.sort()
    n_slice = len(slices)
    
    adata_st_list = []
    for i_slice in range(n_slice):
        adata_st_list.append(adata_light[adata_light.obs[batch_key] == slices[i_slice]].copy())

    for i_slice in range(n_slice - 1):
        print("Spatially register slice", slices[i_slice], "with slice", slices[i_slice+1])

        loc0 = adata_st_list[i_slice].obsm["spatial_regi"]
        loc1 = adata_st_list[i_slice+1].obsm["spatial_regi"]
        
        latent_0 = adata_st_list[i_slice].obsm['latent']
        latent_1 = adata_st_list[i_slice+1].obsm['latent']

        if min(latent_0.shape[0], latent_1.shape[0]) > 30000:   
            n_sample = 20000
        else:
            n_sample = min(latent_0.shape[0], latent_1.shape[0]) // 2
        
        ss_0 = np.random.choice(latent_0.shape[0], size=n_sample, replace=False)
        ss_1 = np.random.choice(latent_1.shape[0], size=n_sample, replace=False)
        
        loc0 = loc0[ss_0, :]
        loc1 = loc1[ss_1, :]
        latent_0 = latent_0[ss_0, :]
        latent_1 = latent_1[ss_1, :]

        mnn_mat = INSPIRE.utils.acquire_pairs(latent_0, latent_1, k=1, metric='euclidean')
        
        rows, cols = np.where(mnn_mat > 0)
        
        if len(rows) == 0:
            continue

        loc0_pair = loc0[rows, :]
        loc1_pair = loc1[cols, :]

        T, R, t = INSPIRE.utils.best_fit_transform(loc1_pair, loc0_pair)
        
        loc1_full = adata_st_list[i_slice+1].obsm["spatial_regi"] 
        loc1_new = np.dot(loc1_full, R.T) + t.T
        adata_st_list[i_slice+1].obsm["spatial_regi"] = loc1_new

    if "spatial_regi" not in adata_full.obsm:
            adata_full.obsm["spatial_regi"] = adata_full.obsm["spatial"].copy()
    for i_slice in range(n_slice):
        indices = adata_st_list[i_slice].obs.index
        coords = adata_st_list[i_slice].obsm["spatial_regi"]
        full_indices = adata_full.obs.index.get_indexer(indices)
        adata_full.obsm["spatial_regi"][full_indices] = coords

    return adata_full

def run_INSPIRE(adata, **args_dict):
    clust = args_dict['clust']
    seed = args_dict['seed']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()

    for ann in ann_list:
        for key in list(ann.obsm.keys()):
            if hasattr(ann.obsm[key], 'values'):
                ann.obsm[key] = ann.obsm[key].values
    
    # Preprocess data
    for ann in ann_list:
        ann.obsm['spatial_tmp'] = ann.obsm['spatial'].copy()
    
    min_cells = min(adata.n_obs for adata in ann_list)
    min_genes = min(adata.n_vars for adata in ann_list)
    # Perform INSPIRE
    rad_coef = 1.1
    if any(ann.n_obs > 10000 for ann in ann_list):  
        adata_st_list, adata_full = prepare_inputs_LGCN(adata_list = ann_list,
                                                        num_hvgs = min(3000, min_genes, min_cells),
                                                        min_genes_qc=1,
                                                        min_cells_qc=1,
                                                        min_concat_dist=100)    
        model = INSPIRE.model.Model_LGCN(adata_st_list=adata_st_list,
                                 n_spatial_factors=20,
                                 n_training_steps=20000,
                                 batch_size=10
                                )
        adata_full, basis_df = model.eval_minibatch(adata_st_list,
                                            adata_full,
                                            batch_size=100
                                           )
        adata = sc.concat(ann_list, join='inner', index_unique=None)
        adata.obsm['latent'] = adata_full.obsm['latent']
    else:
        ann_list, adata = INSPIRE.utils.preprocess(adata_st_list=ann_list,
                                                        num_hvgs=min(3000, min_genes, min_cells),
                                                        min_genes_qc=1,#
                                                        min_cells_qc=1,#
                                                        spot_size=100)
        ann_list = INSPIRE.utils.build_graph_GAT(adata_st_list=ann_list,
                                                    rad_coef=rad_coef)
        model = INSPIRE.model.Model_GAT(adata_st_list=ann_list,
                                        n_spatial_factors=20,
                                        n_training_steps=10000)
        adata, basis_df = model.eval(adata)
    del ann_list
    gc.collect()
    
    # Alignment
    adata = spatial_registration(adata, batch_key='batch')
    adata.obsm['integrated'] = adata.obsm['latent'].copy()
    adata.obsm['spatial_aligned'] = adata.obsm['spatial_regi'].copy()
    adata.obsm['spatial'] = adata.obsm['spatial_tmp'].copy()
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata

