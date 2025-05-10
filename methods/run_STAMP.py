import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np
import sctm
import squidpy as sq
import pandas as pd
import sys
import anndata as ad
import os
import squidpy as sq
import scipy.sparse

current_dir = os.path.dirname(os.path.abspath(__file__)) 
benchmark_dir = os.path.dirname(current_dir)  
sys.path.append(benchmark_dir)
from metrics.utils import *

def reverse_normalization(adata, target_sum=10000,sparse_output=True):
    data = adata.X.copy()
        
    is_sparse = scipy.sparse.issparse(data)
    print(f"Input is sparse: {is_sparse}")
    print(f"Input data shape: {data.shape}")
    print(f"Input data dtype: {data.dtype}")
    
    if is_sparse:
        non_zero_values = data.data
        is_integer = np.all(non_zero_values == np.round(non_zero_values))
    else:
        is_integer = np.all(data == np.round(data))
    
    print(f"All values are integers: {is_integer}")
    
    if is_integer:
        print("Data is already in integer count format. No reverse normalization needed.")
        return adata
    
    print("Data contains non-integer values. Performing reverse normalization...")
    
    if is_sparse:
        data = data.toarray()
        
    if data.max()>20:
        data = data/data.max()*20

    normalized = np.expm1(data)
    normalized_sum = normalized.sum(axis=1)
    total_counts = normalized_sum * (target_sum / normalized_sum.mean())
    adata.obs["estimated_total_counts"] = total_counts
    counts = normalized * total_counts[:, None] / target_sum
    counts = np.ceil(counts).astype(int)
    counts = np.clip(counts, 0, None)
    
    if sparse_output and is_sparse:
        counts = scipy.sparse.csr_matrix(counts)
        print(f"Converted reconstructed counts to sparse CSR format with {counts.nnz} non-zero elements.")

    adata.X = counts
    return adata


def run_STAMP(ann_list, cluster_option):
    for ann in ann_list:
        ann = reverse_normalization(ann)
        
    adata = sc.concat(ann_list, join='inner',index_unique=None)
    
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)
    adata.layers['counts'] = adata.X.copy()
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if adata.shape[1]>3000: 
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
    else:
        adata.var['highly_variable'] = True
    adata.X = adata.layers['counts']
    sq.gr.spatial_neighbors(adata)

    sctm.seed.seed_everything(0)
    model = sctm.stamp.STAMP(
        adata[:, adata.var.highly_variable],
        n_topics = 10,
        # layer="count",
        categorical_covariate_keys=["batch"],
        mode="sgc",
        gene_likelihood="nb")

    model.train(learning_rate = 0.001)
    topic_prop = model.get_cell_by_topic()
    adata.obsm["integrated"] = topic_prop.values
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=cluster_option)
    adata.X = adata.layers['counts']
    return adata

