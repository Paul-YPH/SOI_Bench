import warnings
warnings.filterwarnings("ignore")

import os
import sys

import torch
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA

from GraphST import GraphST

current_dir = os.path.dirname(os.path.abspath(__file__)) 
benchmark_dir = os.path.dirname(current_dir)  
sys.path.append(benchmark_dir)
from metrics.utils import *

#################### Run GraphST ####################
def run_GraphST(ann_list, cluster_option):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.environ['R_HOME'] = '/usr/lib/R'
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    
    adata.layers['counts'] = adata.X.copy()
    # Perform GraphST
    if adata.shape[1] <= 3000:
        adata.var['highly_variable'] = True
    model = GraphST.GraphST(adata, device=device)
    adata = model.train()
    pca = PCA(n_components=min(adata.obsm['emb'].shape[1], 20), random_state=42) 
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding
    adata.obsm['integrated'] = adata.obsm['emb_pca']
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=cluster_option)
    adata.X = adata.layers['counts']
    
    return adata
