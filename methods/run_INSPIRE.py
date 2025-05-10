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

# https://inspire-tutorial.readthedocs.io/en/latest/tutorials/3d_reconstruction/3d_stereoseq_rigid.html

current_dir = os.path.dirname(os.path.abspath(__file__)) 
benchmark_dir = os.path.dirname(current_dir)  
sys.path.append(benchmark_dir)
from metrics.utils import *


### INSPIRE.utils.preprocess no change on obs_names
def preprocess(adata_st_list, # list of spatial transcriptomics anndata objects
               num_hvgs, # number of highly variable genes to be selected for each anndata
               min_genes_qc, # minimum number of genes expressed required for a cell to pass quality control filtering
               min_cells_qc, # minimum number of cells expressed required for a gene to pass quality control filtering
               spot_size, # spot size used in "sc.pl.spatial" for visualization spatial data
               min_concat_dist=50, # minimum distance among data used to re-calculating spatial locations for better visualizations
               limit_num_genes=False, # whether datasets to be integrated only have a limited number of shared genes
              ):
    ## If limit_num_genes=True, get shared genes from datasets before performing any other preprocessing step.
    # Get shared genes
    if limit_num_genes == True:
        print("Get shared genes among all datasets...")
        for i, adata_st in enumerate(adata_st_list):
            if i == 0:
                genes_shared = adata_st.var.index
            else:
                genes_shared = genes_shared & adata_st.var.index
        genes_shared = sorted(list(genes_shared))
        for i, adata_st in enumerate(adata_st_list):
            adata_st_list[i] = adata_st_list[i][:, genes_shared]
        print("Find", str(len(genes_shared)), "shared genes among datasets.")

    
    ## Find shared highly varialbe genes among anndata as features
    print("Finding highly variable genes...")
    for i, adata_st in enumerate(adata_st_list):
        # Remove mt-genes
        adata_st_list[i].var_names_make_unique()
        adata_st_list[i] = adata_st_list[i][:, np.array(~adata_st_list[i].var.index.isna())
                                             & np.array(~adata_st_list[i].var_names.str.startswith("mt-"))
                                             & np.array(~adata_st_list[i].var_names.str.startswith("MT-"))]
        # Remove cells and genes for quality control
        print("shape of adata "+str(i)+" before quality control: ", adata_st_list[i].shape)
        sc.pp.filter_cells(adata_st_list[i], min_genes=min_genes_qc)
        sc.pp.filter_genes(adata_st_list[i], min_cells=min_cells_qc)
        print("shape of adata "+str(i)+" after quality control: ", adata_st_list[i].shape)
        # Find hvgs
        sc.pp.highly_variable_genes(adata_st_list[i], flavor='seurat_v3', n_top_genes=num_hvgs)
        hvgs = adata_st_list[i].var[adata_st_list[i].var.highly_variable == True].sort_values(by="highly_variable_rank").index
        if i == 0:
            hvgs_shared = hvgs
        else:
            hvgs_shared = hvgs_shared & hvgs
        # Add slice label
        adata_st_list[i].obs['slice'] = i
        adata_st_list[i].obs["slice"] = adata_st_list[i].obs["slice"].values.astype(int)
        # Add slice label to barcodes
        # adata_st_list[i].obs.index = adata_st_list[i].obs.index + "-" + str(i)
    hvgs_shared = sorted(list(hvgs_shared))
    print("Find", str(len(hvgs_shared)), "shared highly variable genes among datasets.")

    
    ## Concatenate datasets as a full anndata for better visualization
    print("Concatenate datasets as a full anndata for better visualization...")
    ads = []
    for i, adata_st in enumerate(adata_st_list):
        if i == 0:
            ads.append(adata_st.copy())
        else:
            
            ad_tmp = adata_st.copy()
            xmax_1 = np.max(ads[i-1].obsm["spatial"][:,0])
            xmin_2 = np.min(ad_tmp.obsm["spatial"][:,0])
            ymax_1 = np.max(ads[i-1].obsm["spatial"][:,1])
            ymax_2 = np.max(ad_tmp.obsm["spatial"][:,1])
            ad_tmp.obsm["spatial"][:,0] = ad_tmp.obsm["spatial"][:,0] + (xmax_1 - xmin_2) + min_concat_dist
            ad_tmp.obsm["spatial"][:,1] = ad_tmp.obsm["spatial"][:,1] + (ymax_1 - ymax_2)
            ads.append(ad_tmp.copy())
    del ad_tmp
    adata_full = ad.concat(ads, join="outer")
    sc.pl.spatial(adata_full, spot_size=spot_size)
    del ads


    ## Store counts and library sizes for Poisson modeling, and normalize data for encoder
    print("Store counts and library sizes for Poisson modeling...")
    print("Normalize data...")
    target_sum = 1e4
    if limit_num_genes == True:
        target_sum = 1e3
    for i, adata_st in enumerate(adata_st_list):
        # Store counts and library sizes for Poisson modeling
        st_mtx = adata_st[:, hvgs_shared].X.copy()
        if scipy.sparse.issparse(st_mtx):
            st_mtx = st_mtx.toarray()
        adata_st_list[i].obsm["count"] = st_mtx
        st_library_size = np.sum(st_mtx, axis=1)
        adata_st_list[i].obs["library_size"] = st_library_size
        # Normalize data
        sc.pp.normalize_total(adata_st_list[i], target_sum=target_sum)
        sc.pp.log1p(adata_st_list[i])
        adata_st_list[i] = adata_st_list[i][:, hvgs_shared]
        if scipy.sparse.issparse(adata_st_list[i].X):
            adata_st_list[i].X = adata_st_list[i].X.toarray()

    return adata_st_list, adata_full

def spatial_registration(adata_full,
                         batch_key='batch'
                         ):

    slices = list(set(adata_full.obs[batch_key]))
    n_slice = len(slices)
    
    adata_st_list = []
    for i_slice in range(n_slice):
        adata_st_list.append(adata_full[adata_full.obs[batch_key] == slices[i_slice], :].copy())
        adata_st_list[i_slice].obsm["spatial_regi"] = adata_st_list[i_slice].obsm["spatial"].copy()
    
    angle = 0
    for i_slice in range(n_slice-1):
        print("Spatially register slice", slices[i_slice], "with slice", slices[i_slice+1])

        loc0 = adata_st_list[i_slice].obsm["spatial_regi"]
        loc1 = adata_st_list[i_slice+1].obsm["spatial_regi"]
        
        latent_0 = adata_full[adata_st_list[i_slice].obs.index, :].obsm['latent']
        latent_1 = adata_full[adata_st_list[i_slice+1].obs.index, :].obsm['latent']

        if min(latent_0.shape[0],latent_1.shape[0]) > 30000:   
            n_sample = 20000
        else:
            n_sample = min(latent_0.shape[0],latent_1.shape[0])//2
        
        np.random.seed(1234)
        ss_0 = np.random.choice(latent_0.shape[0], size=n_sample, replace=False)
        ss_1 = np.random.choice(latent_1.shape[0], size=n_sample, replace=False)
        loc0 = loc0[ss_0, :]
        loc1 = loc1[ss_1, :]
        latent_0 = latent_0[ss_0, :]
        latent_1 = latent_1[ss_1, :]

        mnn_mat = INSPIRE.utils.acquire_pairs(latent_0, latent_1, k=1, metric='euclidean')
        idx_0 = []
        idx_1 = []
        for i in range(mnn_mat.shape[0]):
            if np.sum(mnn_mat[i, :]) > 0:
                nns = np.where(mnn_mat[i, :] == 1)[0]
                for j in list(nns):
                    idx_0.append(i)
                    idx_1.append(j)

        loc0_pair = loc0[idx_0, :]
        loc1_pair = loc1[idx_1, :]

        T, R, t = INSPIRE.utils.best_fit_transform(loc1_pair, loc0_pair)
        
        angle = np.arctan2(R[1,0], R[0,0]) * 180 / np.pi
        #adata_st_list[i_slice+1].obs['angle'] = angle

        loc1 = adata_st_list[i_slice+1].obsm["spatial_regi"] 
        loc1_new = np.dot(loc1, R.T) + t.T
        adata_st_list[i_slice+1].obsm["spatial_regi"] = loc1_new

    #adata_full.obs['angle'] = angle
    return adata_st_list

#################### Run INSPIRE ####################
def run_INSPIRE(ann_list, cluster_option):
    # Preprocess data

    for ann in ann_list:
        ann.obsm['spatial_tmp'] = ann.obsm['spatial'].copy()
        
    ann_list, adata = preprocess(adata_st_list=ann_list,
                                                    num_hvgs=3000,
                                                    min_genes_qc=0,#
                                                    min_cells_qc=0,#
                                                    spot_size=100)
    # Perform INSPIRE
    ann_list = INSPIRE.utils.build_graph_GAT(adata_st_list=ann_list,
                                                rad_coef=1.1)
    model = INSPIRE.model.Model_GAT(adata_st_list=ann_list,
                                    n_spatial_factors=20,
                                    n_training_steps=10000,
                                )
    adata, basis_df = model.eval(adata)
    # Alignment
    tmp_list = spatial_registration(adata, batch_key='batch')
    adata = sc.concat(tmp_list, join='inner', index_unique=None)
    adata.obsm['integrated'] = adata.obsm['latent']
    adata.obsm['spatial_aligned'] = adata.obsm['spatial_regi']
    adata.obsm['spatial'] = adata.obsm['spatial_tmp']

    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=cluster_option)
    
    if 'high_quality_transfer' in adata.obs.columns:
        adata.obs['high_quality_transfer'] = adata.obs['high_quality_transfer'].astype(str)


    return adata

