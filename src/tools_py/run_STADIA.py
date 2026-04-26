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

def run_STADIA(adata, **args_dict):
    seed = args_dict['seed']
    
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
        adata.X = adata.X.toarray()
        
    r_adata = anndata2ri.py2rpy(adata)
    r.assign("adata_r", r_adata)

    r.assign("py_seed", seed)

    r("""
    options(Seurat.object.assay.version = "v3")

    library(stadia)
    library(Seurat)
    library(mclust)
    library(SingleCellExperiment)
    library(BiocNeighbors)

    meta_data <- as.data.frame(colData(adata_r))
    counts_matrix <- assays(adata_r)$X

    new_InitialC <- function(Y, q = 7) {
    init <- list()

    clust_result <- tryCatch({
        mclust::Mclust(data = Y, G = q, modelNames = "EEE", verbose = FALSE)
    }, error = function(e) NULL)

    if (is.null(clust_result) || is.null(clust_result$parameters)) {
        message("EEE failed, trying EEI selection...")
        clust_result <- tryCatch({
        mclust::Mclust(data = Y, G = q, modelNames = "EEI", verbose = FALSE)
        }, error = function(e) NULL)
    }

    if (is.null(clust_result) || is.null(clust_result$parameters)) {
        message("EEI failed, trying EII selection...")
        clust_result <- tryCatch({
        mclust::Mclust(data = Y, G = q, modelNames = "EII", verbose = FALSE)
        }, error = function(e) NULL)
    }

    if (is.null(clust_result) || is.null(clust_result$parameters)) {
        message(paste0("Warning: All Mclust models failed for G=", q, ". Falling back to K-means."))
        km <- kmeans(Y, centers = q, iter.max = 100, nstart = 25)
        init$c <- km$cluster
        init$mu <- t(km$centers)
        p <- ncol(Y)
        global_var <- apply(Y, 2, var)
        global_var[global_var < 1e-6] <- 1e-6
        global_var[is.na(global_var)] <- 1e-6
        init$Lambda <- diag(global_var)
        
    } else {
        init$c <- clust_result$classification
        init$mu <- clust_result$parameters$mean
        sigma <- clust_result$parameters$variance$Sigma
        
        if (length(dim(sigma)) == 3) {
        init$Lambda <- apply(sigma, c(1, 2), mean)
        } else {
        init$Lambda <- sigma
        }
    }
    return(init)
    }
    assignInNamespace(".InitialC", new_InitialC, ns = "stadia")
    
    new_InitiParameters <- function(object.list, k = 7, d = 35) {
        result <- list()
        batch_vec <- rep(seq_along(object.list), times = sapply(object.list, ncol))
        n <- sum(sapply(object.list, ncol))
        p <- sapply(object.list, nrow)[1]
        b <- length(object.list)
        if (b == 1) {
            data2use <- t(as.matrix(GetAssayData(object.list[[1]], layer = "scale.data")))
        } else {
            data_use_list <- sapply(object.list, function(object) {
                return(as.matrix(Seurat::GetAssayData(object, layer = "scale.data")))
            }, simplify = FALSE)
            data2use <- t(Reduce(cbind, data_use_list))
        }
        data2use[!is.finite(data2use)] <- 0
        batchVec2M <- getFromNamespace("batchVec2M", "stadia")
        M <- batchVec2M(batch_vec)
        fm <- lm(data2use ~ M + 0)
        result$B <- t(fm$coefficients)
        data2use <- fm$residuals
        pca_result <- irlba::irlba(data2use, nv = d)
        result$L <- pca_result$v %*% diag(pca_result$d)
        result$F <- pca_result$u
        varimax_L <- varimax(result$L)
        result$L <- varimax_L$loadings
        result$F <- result$F %*% varimax_L$rotmat
        data2use <- data2use - result$F %*% t(result$L)
        result$T <- sapply(1:b, function(i) {
            apply(data2use[batch_vec == i, , drop=FALSE] * data2use[batch_vec == i, , drop=FALSE], 2, sum) / (nrow(object.list[[i]]) - 1)
        })
        result$T <- 1 / result$T
        clust_result <- stadia:::.InitialC(result$F, q = k)
        result$c <- clust_result$c
        result$mu <- clust_result$mu
        result$Lambda <- solve(clust_result$Lambda)
        result$omega <- rep(1, n)
        result$p <- rep(0.5, d)
        result$gamma <- matrix(rbinom(p * d, 1, 0.5), ncol = d)
        return(result)
    }

    stadia_fixed <- function(object.list, hyper, dim = 35, n_cluster = 7,
                            platform = c("visium", "st", "others"),
                            adj.cutoff = 50, icm.maxiter = 10, em.maxiter = 30,
                            min.features = 200, min.spots = 20, nfeatures = 2000,
                            verbose = TRUE, verbose.in = TRUE, ncores = NULL) {

    d <- dim
    K <- n_cluster

    cat("Preprocess data...\n")
    object.list <- PreprocessData(object.list, 
                                    min.features = min.features, 
                                    min.spots = min.spots, 
                                    nfeatures = nfeatures)

    cat("Set initialization...\n")
    init <- new_InitiParameters(object.list, K, d)

    batch_vec <- rep(seq_along(object.list), times = sapply(object.list, ncol))
    if (is.null(ncores)) {
        ncores <- max(1, round(parallel::detectCores() * 0.8))
    }

    position <- Reduce(rbind, lapply(object.list, function(x) {
        cbind(x$row, x$col)
    }))

    X.list <- lapply(object.list, function(x) {
        Seurat::GetAssayData(x, layer = "scale.data")
    })
    batch.list <- lapply(X.list, t)
    X <- do.call(cbind, X.list)

    if (!is.null(platform) && substr(tolower(platform), 1, 1) %in% c("v", "s")) {
        platform <- tolower(platform)
        platform <- match.arg(platform)
    } else {
        platform <- "others"
    }

    adj_mat_mnn <- mnn_adjacent(batch.list, d = d)

    out <- stadia:::stadia_EM_SP(
        X, position, batch_vec, adj_mat_mnn,
        hyper, init, platform,
        d, K, adj.cutoff, icm.maxiter, em.maxiter, 
        verbose = verbose, ncores = ncores
    )

    rownames(out$L) <- rownames(X)
    colnames(out$factors) <- colnames(X)

    return(out)
    }

    meta_data <- as.data.frame(colData(adata_r))
    counts_matrix <- assays(adata_r)$X

    set.seed(py_seed)

    seuList <- list()
    batches <- unique(meta_data$batch)

    for (i in seq_along(batches)) {
    batch_name <- batches[i]
    batch_idx <- meta_data$batch == batches[i]
    batch_meta <- meta_data[batch_idx, ]
    batch_counts <- counts_matrix[, batch_idx]

    cat("Processing batch:", batch_name, "with", sum(batch_idx), "cells\n")

    seu <- CreateSeuratObject(counts = batch_counts, meta.data = batch_meta)
    seu@assays$RNA <- as(seu@assays$RNA, "Assay")
    seu <- NormalizeData(seu, verbose = FALSE)
    seuList[[batch_name]] <- seu
    }

    d <- 35
    for (batch in seuList) {
    min_dim <- min(nrow(batch) - 1, ncol(batch) - 1)
    d <- min(d, min_dim)
    cat("Batch dimensions: ", nrow(batch), "x", ncol(batch), 
        ", max possible d:", min_dim, "\n")
    }
    cat("Final d:", d, "\n\n")
    K <- length(unique(meta_data$type))
    cat("K (from cell types):", K, "\n")

    etas <- 0.15
    unique_techs <- unique(meta_data$technology)
    technique <- "others"
    if (length(unique_techs) == 1) {
    tech_lower <- tolower(unique_techs[1])
    if (grepl("visium", tech_lower)) {
        technique <- "visium"
    } else if (grepl("st", tech_lower)) {
        technique <- "st"
    } else if (grepl("stereoseq", tech_lower)) {
        technique <- "visium"
    } else if (grepl("merscope", tech_lower)) {
        technique <- "visium"
        etas <- 0.1
    } 
    }
    hyper <- HyperParameters(seuList, d = d, eta = etas)
    cat("Hyperparameters set with d =", d, "and eta =", etas, "\n\n")

    
    nfeatures_value <- min(2000, nrow(counts_matrix), ncol(counts_matrix))

    obj_stadia <- stadia_fixed(
        seuList, 
        hyper, 
        dim = d,
        n_cluster = K,
        platform = technique, 
        em.maxiter = 30,
        icm.maxiter = 10,
        min.features = 3, 
        min.spots = 3, 
        nfeatures = nfeatures_value,
        ncores = 1
    )

    embeddings <- as.data.frame(t(obj_stadia[["factors"]]))
    rownames(embeddings) <- colnames(obj_stadia[["factors"]])
    clusters <- obj_stadia[["c_vec"]]
    cluster_result <- data.frame(cluster = clusters,
                                row.names = colnames(obj_stadia[["factors"]]))
                                
    gc()

    """)

    emb = r('embeddings')
    embeddings_df = pd.DataFrame(emb, index=r('rownames(embeddings)'))  
    common_cells = adata.obs_names.intersection(embeddings_df.index)
    adata = adata[common_cells,:]
    embeddings_df = embeddings_df.loc[common_cells]
    adata.obsm['integrated'] = embeddings_df.values
    cluster_result = r('cluster_result')
    cluster_result = pd.DataFrame(cluster_result, index=r('rownames(cluster_result)'))
    cluster_result = cluster_result.loc[common_cells]
    adata.obs['benchmark_cluster'] = cluster_result.iloc[:, 0].values
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata
