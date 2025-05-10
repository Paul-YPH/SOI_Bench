import warnings
warnings.filterwarnings("ignore")

import os
import sys
import yaml

import scanpy as sc
import os
import torch
from sklearn.decomposition import PCA

import STG3Net as MODEL


def mapping2int(string_array):
    mapping = {}
    result = []
    for string in string_array:
        if string not in mapping:
            mapping[string] = len(mapping)
        result.append(mapping[string])
    return result, mapping

os.environ['R_HOME'] = '/usr/lib/R'
os.environ['R_USER'] = '/net/mulan/home/penghuy/anaconda3/envs/stg3net/lib/python3.10/site-packages/rpy2'

current_dir = os.path.dirname(os.path.abspath(__file__)) 


with open(os.path.join(current_dir, 'Config', 'Config.yaml'), 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

current_dir = os.path.dirname(os.path.abspath(__file__)) 
benchmark_dir = os.path.dirname(current_dir)  
sys.path.append(benchmark_dir)
from metrics.utils import *

#################### Run STG3Net ####################
def run_STG3Net(ann_list, sample_list, cluster_option):
    for ann in ann_list:
        sample = ann.obs['batch'].unique()[0]
        ann.obs['batch_name'] = ann.obs['batch']
        batch_unique_values = ann.obs['batch']
        mapped_result, mapping = mapping2int(batch_unique_values)
        ann.obs['slice_id'] = mapped_result
        graph_dict_tmp = MODEL.graph_construction(ann, config['data']['k_cutoff'])
        if sample == sample_list[0]:
            adata = ann
            graph_dict = graph_dict_tmp
            name = sample
            adata.obs['proj_name'] = sample
        else:
            var_names = adata.var_names.intersection(ann.var_names)
            adata = adata[:, var_names]
            ann = ann[:, var_names]
            ann.obs['proj_name'] = sample

            adata = adata.concatenate(ann)
            graph_dict = MODEL.combine_graph_dict(graph_dict, graph_dict_tmp)
            name = name + '_' + sample
    
    import scipy.sparse
    adata.layers['count'] = adata.X.toarray() if isinstance(adata.X, scipy.sparse.spmatrix) else adata.X
    # sc.pp.filter_genes(adata, min_cells=50)
    # sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    if adata.shape[1]>3000: 
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    else:
        adata.var['highly_variable'] = True
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    adata_X = PCA(n_components=min(200, adata.shape[1]), random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Train STG3Net
    print('### Performing STG3Net...')
    net = MODEL.G3net(adata, graph_dict=graph_dict, device=device, config=config, num_cluster=adata.obs['Ground Truth'].nunique())
    net.train(verbose=1, method = 'kmeans')###
    enc_rep, recon = net.process()
    enc_rep = enc_rep.data.cpu().numpy()
    recon = recon.data.cpu().numpy()
    adata.obsm['integrated'] = enc_rep
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=cluster_option)
    adata.X = adata.layers['count']
    return adata    