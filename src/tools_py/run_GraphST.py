import warnings
warnings.filterwarnings("ignore")

import torch
import scanpy as sc
import gc
from GraphST import GraphST
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering

def run_GraphST(adata, **args_dict):
    clust = args_dict['clust']
    seed = args_dict['seed']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    del ann_list
    gc.collect()
    
    # Perform GraphST
    if adata.obs['technology'].unique() == ['stereoseq']:
        datatype = "Stereo-seq"
    elif adata.obs['technology'].unique() == ['slideseq']:
        datatype = "Slide"
    else:
        datatype = "10X"
    model = GraphST.GraphST(adata, device=device, datatype = datatype)
    adata = model.train()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=20, random_state=seed) 
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['integrated'] = embedding

    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)    
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata

