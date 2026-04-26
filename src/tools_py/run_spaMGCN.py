import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import scanpy as sc
import torch
from torch.backends import cudnn
import argparse
import gc
try:
    import yaml
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', '--user', 'pyyaml', '-q'], check=False)
from spamgcn.utils.misc import *
from spamgcn.train.train3 import Train,Test
from spamgcn.model.Creat_model import creat_model
from spamgcn.utils.preprocess import construct_graph_by_coordinate, transform_adjacent_matrix, preprocess_graph
from utils import get_ann_list, create_lightweight_adata

import spamgcn
from clustering import mclust_R_v2
spamgcn.train.utils.mclust_R = mclust_R_v2

def build_args(): # config file
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default="acm")
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--n_input', type=int, default=100)
    parser.add_argument('--n_z', type=int, default=20)
    parser.add_argument('--freedom_degree', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=700)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--sigma', type=float, default=0.7)
    parser.add_argument('--loss_n', type=float, default=0.01)
    parser.add_argument('--loss_w', type=float, default=0.1)
    parser.add_argument('--loss_s', type=float, default=0.1)
    parser.add_argument('--loss_a', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--acc', type=float, default=-1)
    parser.add_argument('--f1', type=float, default=-1)
    args = parser.parse_args([])  
    return args

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
def pca(adata, use_reps=None, n_comps=10):
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray()) 
        else:   
            feat_pca = pca.fit_transform(adata.X)
    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):
    import numpy as np
    import scipy
    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)
    if not inplace:
        adata = adata.copy()
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     

def create_adj(adata, knn):
    cell_position_omics1 = adata.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=knn)
    adata.uns['adj_spatial'] = adj_omics1
    adj_spatial_omics1 = adata.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics1 = adj_spatial_omics1.toarray()
    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1>1, 1, adj_spatial_omics1)
    adj = preprocess_graph(adj_spatial_omics1)
    return adj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_spaMGCN(adata, **args_dict):
    args = build_args()
    seed = args_dict['seed']
    clust = args_dict['clust']
    knn = args_dict['knn']
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    if 'gam' in config['sample_list'].tolist():
        adata_omics1 = ann_list[config['sample_list'].tolist().index('rna')]
        adata_omics2 = ann_list[config['sample_list'].tolist().index('gam')]
        # ATAC
        sc.pp.filter_genes(adata_omics2, min_cells=1)
        sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)# article: 3000
        sc.pp.normalize_total(adata_omics2, target_sum=1e4)
        sc.pp.log1p(adata_omics2)
        sc.pp.scale(adata_omics2)
    elif 'adt' in config['sample_list'].tolist():
        # ADT
        adata_omics1 = ann_list[config['sample_list'].tolist().index('rna')]
        adata_omics2 = ann_list[config['sample_list'].tolist().index('adt')]
        clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
    
    del ann_list
    gc.collect()

    adata_omics1.obs_names = [n.split('_', 1)[1] for n in adata_omics1.obs_names]
    adata_omics2.obs_names = [n.split('_', 1)[1] for n in adata_omics2.obs_names]

    # RNA
    sc.pp.filter_genes(adata_omics1, min_cells=1)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)# article: 3000
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)
    
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
    
    n_comps = min(30,adata_omics2.shape[1]-1,adata_omics2.shape[0]-1)
    adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=n_comps)
    adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_comps)
    
    args.n_input=n_comps
    args.n_input1=n_comps
    args.n_clusters=len(adata_omics2.obs['Ground Truth'].unique())
    args.random_seed=seed
    fix_seed(seed)
    
    args.loss_n=0.01
    args.lr=0.01
    args.sigma=0.5
    device='cuda:0'
    args.tool=clust
    args.name=adata_omics1.obs['batch'].unique()[0]
    
    label=adata_omics2.obs['Ground Truth'].values

    args.n_clusters1=len(adata_omics1.obs['Ground Truth'].unique())
    args.n_clusters2=len(adata_omics2.obs['Ground Truth'].unique())
    adj_train=create_adj(adata_omics1, knn)
    adj_train = adj_train.to(device)    
    features_omics1 = torch.FloatTensor(adata_omics1.obsm['feat'].copy()).to(device)
    features_omics2 = torch.FloatTensor(adata_omics2.obsm['feat'].copy()).to(device)
    model = creat_model('spamgcn', args).to(device)
    model=Train(80, model, adata_omics1,features_omics1,features_omics2, adj_train, label, device, args)
    nmi, ari, ami, homogeneity, completeness, v_measure=Test(model,adata_omics1,features_omics1,features_omics2,adj_train,label,device,args,clust)
    
    adata = adata_omics1.copy()
    del adata_omics1, adata_omics2
    adata.obsm['integrated'] = adata.obsm['MGCN']
    adata.obs['benchmark_cluster'] = adata.obs['pred']
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata