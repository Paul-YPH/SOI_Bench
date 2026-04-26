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

def run_PRECAST(adata, **args_dict):
    seed = args_dict['seed']
    
    ann_list, config = get_ann_list(adata)
    del adata
    anndata2ri.activate()
    pandas2ri.activate()
    for ann in ann_list:
        ann.obs['type'] = ann.obs['Ground Truth']
        if ann.obs['array_row'].isna().any():
            ann.obs['col'] = ann.obsm['spatial'][:, 0].astype(np.float32)
            ann.obs['row'] = ann.obsm['spatial'][:, 1].astype(np.float32)
        else:
            ann.obs['row'] = ann.obs['array_row'].astype(np.float32)
            ann.obs['col'] = ann.obs['array_col'].astype(np.float32)
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
    r("""
    library(Seurat)
    library(SingleCellExperiment) 
    library(PRECAST)

    set.seed(py_seed)
    
    meta_data <- as.data.frame(colData(adata_r))
    counts_matrix <- assays(adata_r)$X
    rm(adata_r)
    gc()

    seuList <- list()
    batches <- unique(meta_data$batch)
    for(i in seq_along(batches)) {
        batch_idx <- meta_data$batch == batches[i]
        batch_meta <- meta_data[batch_idx,]
        batch_counts <- counts_matrix[,batch_idx]
        seu <- CreateSeuratObject(counts = batch_counts, meta.data = batch_meta)
        seuList[[i]] <- seu
    }
    rm(counts_matrix)
    gc()

    metadataList <- lapply(seuList, function(x) x@meta.data)
    for (r in seq_along(metadataList)) {
        meta_data <- metadataList[[r]]
        cat('Batch', r, 'spatial coordinates check:', 
            all(c('row', 'col') %in% colnames(meta_data)), '\n')
    }
    
    default_gene_number = 3000
    if (nrow(batch_counts) <= default_gene_number) {
        preobj <- CreatePRECASTObject(
            seuList = seuList, 
            selectGenesMethod = 'HVGs', 
            customGenelist = rownames(batch_counts),
            premin.spots = 1,  
            premin.features=1, 
            postmin.spots=1, 
            postmin.features=1
        )
    } else {
        preobj <- CreatePRECASTObject(
            seuList = seuList, 
            selectGenesMethod = 'HVGs', 
            gene.number = default_gene_number,
            premin.spots = 1,  
            premin.features=1, 
            postmin.spots=1, 
            postmin.features=1
        )
    }
    rm(seu_list)
    invisible(gc())

    if (length(unique(meta_data$technology)) == 1) {
        if (unique(meta_data$technology) == 'st') {
            platform = 'ST'
        } else {
            platform = 'Visium'
        }
    } else {
        platform = 'Visium'
    }
    PRECASTObj <- AddAdjList(preobj, platform = platform)
    PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = TRUE, coreNum = 1, maxIter = 30, verbose = TRUE)
    PRECASTObj <- PRECAST(PRECASTObj, K = length(unique(meta_data$type)))
    PRECASTObj <- SelectModel(PRECASTObj)

    if (unique(meta_data$species) == 'Mouse') {
        seuInt <- IntegrateSpaData(PRECASTObj, species = 'Mouse')
    } else {
        seuInt <- IntegrateSpaData(PRECASTObj, species = 'Human')
    }

    emb=list()
    for (batch_name in unique(seuInt$batch)) {
    emb[[batch_name]]<-as.data.frame(seuInt@reductions$PRECAST@cell.embeddings[seuInt$batch==batch_name,])
    
    }
    embeddings = do.call(rbind, emb)
    rownames(embeddings) <- unlist(lapply(emb, rownames))

    cluster_result<-as.data.frame(seuInt$cluster)
    rownames(cluster_result) <- colnames(seuInt)

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
