import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import scanpy as sc
import os
import torch
from sklearn.decomposition import PCA
import STG3Net as MODEL
import scipy.sparse 
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering

def mapping2int(string_array):
    mapping = {}
    result = []
    for string in string_array:
        if string not in mapping:
            mapping[string] = len(mapping)
        result.append(mapping[string])
    return result, mapping

def run_STG3Net(adata, config_dir, **args_dict):
    sample_list = adata.uns['config']['sample_list']
    seed = args_dict['seed']
    knn = args_dict['knn']
    pcs = args_dict['pcs']
    set_seed(seed)
    clust = args_dict['clust']
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    with open(os.path.join(config_dir, 'Config.yaml'), 'r', encoding='utf-8') as f:
        config_tool = yaml.load(f.read(), Loader=yaml.FullLoader)
    for ann in ann_list:
        sample = ann.obs['batch'].unique()[0]
        ann.obs['batch_name'] = ann.obs['batch']
        batch_unique_values = ann.obs['batch']
        mapped_result, mapping = mapping2int(batch_unique_values)
        ann.obs['slice_id'] = mapped_result
        graph_dict_tmp = MODEL.graph_construction(ann, n = knn)
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
            
    del ann_list
    gc.collect()
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=min(3000, adata.shape[1],adata.shape[0]))
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    adata_X = PCA(n_components=min(pcs, adata.shape[1]-1,adata.shape[0]-1)).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Train STG3Net
    print('### Performing STG3Net...')
    net = MODEL.G3net(adata, graph_dict=graph_dict, device=device, config=config_tool, num_cluster=adata.obs['Ground Truth'].nunique())
    net.train(verbose=1, method = 'kmeans')###
    enc_rep, recon = net.process()
    enc_rep = enc_rep.data.cpu().numpy()
    recon = recon.data.cpu().numpy()
    adata.obsm['integrated'] = enc_rep
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)    
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata    