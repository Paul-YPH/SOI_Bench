import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import anndata2ri
from rpy2 import robjects
from rpy2.robjects import r, pandas2ri
from scipy.sparse import issparse
import pandas as pd
import numpy as np  
import gc
from utils import get_ann_list, create_lightweight_adata

def run_BANKSY(adata, **args_dict):
    
    seed = args_dict['seed']
    knn = args_dict['knn']
    
    ann_list, config = get_ann_list(adata)
    del adata
    anndata2ri.activate()
    pandas2ri.activate()
    for ann in ann_list:
        ann.obs['type'] = ann.obs['Ground Truth']
        ann.obs['X'] = ann.obsm['spatial'][:, 0].astype(np.float32)
        ann.obs['Y'] = ann.obsm['spatial'][:, 1].astype(np.float32)
        ann.obs = ann.obs[['type', 'X', 'Y', 'batch', 'Ground Truth', 'technology']]
        for key in list(ann.obsm.keys()):
            if key != 'spatial':
                del ann.obsm[key]

    adata = sc.concat(ann_list, join='inner', index_unique=None)
    clust_cols = [col for col in adata.obs.columns if col.startswith('clust_')]
    adata.obs.drop(columns=clust_cols, inplace=True)
    del ann_list
    gc.collect()
    
    tech_type = adata.obs['technology'].unique()
    if 'visium' in tech_type:
        lambda_val = 0.2
    else:
        lambda_val = 0.8

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
    
    r.assign("py_knn", knn)
    r.assign("py_seed", seed)
    r.assign("py_lambda", lambda_val)
    
    r("""
    library(SummarizedExperiment)
    library(SpatialExperiment)
    library(scuttle)
    library(Banksy)
    library(Seurat)
    
    set.seed(py_seed)
    lambda <- py_lambda

    meta_data <- as.data.frame(colData(adata_r))
    counts_matrix <- assays(adata_r)$X
    rm(adata_r)
    gc()

    spe_list <- list()
    batches <- unique(meta_data$batch)
    for(i in seq_along(batches)) {
        batch_idx <- meta_data$batch == batches[i]
        batch_meta <- meta_data[batch_idx,]
        batch_counts <- counts_matrix[,batch_idx]
        sce <- SingleCellExperiment(assays = list(counts = as.matrix(batch_counts)))
        colData(sce) <- DataFrame(batch_meta)
        spe_list[[i]] <- sce
    }
    rm(counts_matrix)
    gc()
    
    seu_list <- lapply(spe_list, function(x) {
        x <- as.Seurat(x, data = NULL)
        NormalizeData(x, scale.factor = 5000, normalization.method = 'RC')
    })

    hvgs <- lapply(seu_list, function(x) {
        n_genes <- nrow(x)
        n_features <- min(2000, n_genes)
        VariableFeatures(FindVariableFeatures(x, nfeatures = n_features))
    })
    hvgs <- Reduce(union, hvgs)

    aname <- "normcounts"
    spe_list <- Map(function(spe, seu) {
        assay(spe, aname, withDimnames=FALSE) <- GetAssayData(seu)
        common_hvgs <- intersect(hvgs, rownames(spe))
        spe[common_hvgs,]
    }, spe_list, seu_list)
    rm(seu_list)
    invisible(gc())

    compute_agf <- FALSE
    k_geom <- py_knn
    spe_list <- lapply(spe_list, computeBanksy, 
                    assay_name = aname,
                    compute_agf = compute_agf, 
                    k_geom = k_geom,
                    coord_names = c("X", "Y"))

    spe_joint <- do.call(cbind, spe_list)
    rm(spe_list)
    invisible(gc())

    use_agf <- FALSE
    spe_joint <- runBanksyPCA(spe_joint, use_agf = use_agf, lambda = lambda, group = "batch", seed = py_seed)
    spe_joint <- clusterBanksy(spe_joint, use_agf = use_agf, algo="mclust", lambda = lambda, mclust.G = length(unique(meta_data$type)), seed = py_seed)
    
    embeddings <- as.data.frame(reducedDims(spe_joint)[[reducedDimNames(spe_joint)]])
    rownames(embeddings) <- colnames(spe_joint)

    clust_col <- grep("^clust_", colnames(colData(spe_joint)), value = TRUE)[1]
    clusters <- colData(spe_joint)[[clust_col]]
    cluster_result <- data.frame(cluster = clusters, row.names = colnames(spe_joint))
    rm(spe_joint)
    gc()
    """)

    emb = r('embeddings')
    embeddings_df = pd.DataFrame(emb, index=r('rownames(embeddings)'))
    r('rm(embeddings)'); r('gc()')
    
    common_cells = obs_names.intersection(embeddings_df.index)
    adata = adata[common_cells, :]
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