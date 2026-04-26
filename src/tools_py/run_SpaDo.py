import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import anndata2ri
from rpy2.robjects import r, pandas2ri
from scipy.sparse import issparse
import pandas as pd
import numpy as np
import gc
from utils import get_ann_list, create_lightweight_adata

def run_SpaDo(adata, **args_dict):
    seed = args_dict['seed']
    knn = args_dict['knn']
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    anndata2ri.activate()
    pandas2ri.activate()
    for ann in ann_list:
        ann.obs['type'] = ann.obs['Ground Truth']
        ann.obs['col'] = ann.obsm['spatial'][:, 0].astype(np.float32)
        ann.obs['row'] = ann.obsm['spatial'][:, 1].astype(np.float32)
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    
    del ann_list
    gc.collect()
    
    if issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = adata.X.astype(np.float32)
        
    r_adata = anndata2ri.py2rpy(adata)
    r.assign("adata_r", r_adata)
    obs_names = adata.obs_names.copy()
    del r_adata
    adata.X = None
    gc.collect()

    r.assign("py_seed", seed)
    r.assign("knn", knn)
    r("""
    library(SpaDo)
    library(parallel)
    library(Seurat)

    library(future)
    plan(sequential)
    message("Enforcing sequential processing for stability in rpy2 container.")

    meta_data <- as.data.frame(colData(adata_r))
    counts_matrix <- assays(adata_r)$X
    rm(adata_r)
    gc()

    set.seed(py_seed)

    slice_choose<-unique(meta_data$batch)

    coordinate_list = list()
    for (slice_name in slice_choose) {
        coordinate_list[[slice_name]] = meta_data[meta_data$batch == slice_name, c("row", "col")]
    }

    unique_techs <- unique(meta_data$technology)
    if (any(grepl("osmfish", unique_techs, ignore.case = TRUE))) {
        technique <- "osmFISH"
    } else if (any(unique_techs == "merfish")) {
        technique <- "merFISH"  
    } else if (any(unique_techs == "starmap")) {
        technique <- "STARmap"
    } else {
        technique <- "Others"
    }
    
    cluster_list = list()
    for (slice_name in slice_choose) {
    exp<-counts_matrix[,rownames(coordinate_list[[slice_name]])]

    exp_normalized<-SpatialNormalize(expression_profile = exp,ST_method = technique)
    initial_result<-InitialClustering(expression_profile = exp_normalized,user_offered = F)
    cluster_list[[slice_name]] <- initial_result$sample_information
    }
    cluster_list<-cluster_list[slice_choose]

    cell_type_distribution_multiple<-SpatialCellTypeDistribution_multiple(
    sample_information_coordinate_list  = coordinate_list,
    sample_information_cellType_list = cluster_list,
    sequence_resolution = "single_cell",
    k = knn)

    tmp<-cell_type_distribution_multiple[["datasets_lable"]]
    current_names <- names(tmp)
    for (slice_name in slice_choose) {
        pattern <- paste0("^", slice_name, "_")
        current_names <- sub(pattern, "", current_names)
    }
    names(tmp) <- current_names

    ### embedding
    cell_type_distribution<-cell_type_distribution_multiple[["cell_type_distribution_combine"]]

    ### embedding
    cell_type_distribution<-cell_type_distribution_multiple[["cell_type_distribution_combine"]]
    DD<-DistributionDistance(cell_type_distribution,distance = "JSD",no_cores=1)
    pca_result <- prcomp(DD,center = TRUE, scale. = TRUE)
    embeddings<-pca_result$x[, 1:50]
    embeddings<-as.data.frame(embeddings)
    colnames(embeddings) <- paste('PC', 1:50, sep = '')
    rownames(embeddings) <- names(tmp)

    library(fastcluster)
    num = length(unique(meta_data$type))
    domain_hclust<-DomainHclust(distribution_distance = DD,autoselection = FALSE,domain_num = num)
    clusters<-domain_hclust$hclust_result_df[[paste('Domain_level',num,sep = "_")]]
    clusters<-as.factor(clusters)
    clusters<-as.numeric(clusters)
    cluster_result <- data.frame(cluster = clusters,
                                row.names = names(tmp))
    gc()
    """)

    emb = r('embeddings')
    embeddings_df = pd.DataFrame(emb, index=r('rownames(embeddings)'))
    r('rm(embeddings)'); r('gc()')

    common_cells = adata.obs_names.intersection(embeddings_df.index)
    adata = adata[common_cells,:]
    embeddings_df = embeddings_df.loc[common_cells]
    adata.obsm['integrated'] = embeddings_df.values
    del embeddings_df

    cluster_result = r('cluster_result')
    cluster_result = pd.DataFrame(cluster_result, index=r('rownames(cluster_result)'))
    r('rm(cluster_result)'); r('gc()')

    cluster_result = cluster_result.loc[common_cells]
    adata.obs['benchmark_cluster'] = cluster_result.iloc[:, 0].values
    del cluster_result
    gc.collect()
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata
