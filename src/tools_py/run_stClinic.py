import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import random
import torch
import numpy as np
import stClinic as stClinic
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering

#Set parameters
used_device   =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = stClinic.parameter_setting()
args = parser.parse_args()
args.n_top_genes = 3000
args.lr_prediction = 0.01
args.lr_integration = 0.00005

def run_stClinic(adata, **args_dict):
    seed = args_dict['seed']
    pcs = args_dict['pcs']
    knn = args_dict['knn']
    set_seed(seed)
    clust = args_dict['clust']
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    adj_list    = []
    section_ids = []
    for ann in ann_list:
        section_id = ann.obs['batch'].unique()[0]
        # ann.var_names_make_unique(join="++")
        stClinic.Cal_Spatial_Net(ann, k_cutoff=knn, model='KNN')
        # Normalization
        sc.pp.highly_variable_genes(ann, flavor="seurat_v3", n_top_genes=args.n_top_genes)
        sc.pp.normalize_total(ann, target_sum=1e4)
        sc.pp.log1p(ann)
        ann = ann[:, ann.var['highly_variable']]
        sc.tl.pca(ann, n_comps=min(pcs, ann.shape[1]-1, ann.shape[0]-1), random_state=seed)
        adj_list.append(ann.uns['adj'])
        section_ids.append(section_id)
    # Concat scanpy objects
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    del ann_list
    gc.collect()
    
    if 'data_id' in adata.obs.columns:
        data_id = adata.obs['data_id'].unique()[0]
        if data_id in ['SD30','SD31','SD32','SD33','SD34','SD35','SD42']:
            args.lr_integration = 0.00001
    
    adj_concat = stClinic.inter_linked_graph(adj_list, section_ids, mnn_dict=None)
    adata.uns['adj']      = adj_concat
    adata.uns['edgeList'] = np.nonzero(adj_concat)
    centroids_num = adata.obs['Ground Truth'].nunique()
    adata.obs['Ground Truth'] = adata.obs['Ground Truth'].astype('category')
    adata.obs["batch"] = adata.obs["batch"].astype('category')
    del adj_list
    gc.collect()
    torch.cuda.empty_cache()
    
    adata  = stClinic.train_stClinic_model(adata, n_centroids=centroids_num, lr=args.lr_integration, device=used_device,batch_name= "batch")
    adata.obsm['integrated'] = adata.obsm['stClinic']

    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata