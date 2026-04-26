import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA 
import harmonypy as hm
import scipy
import SEDR
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering

def run_SEDR(adata, **args_dict):
    seed = args_dict['seed']
    knn = args_dict['knn']
    pcs = args_dict['pcs']
    clust = args_dict['clust']
    set_seed(seed)
    
    sample_list = adata.uns['config']['sample_list']
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    for ann in ann_list:
        graph_dict_tmp = SEDR.graph_construction(ann, knn)

        sample = ann.obs['batch'].unique()[0]
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

            adata = adata.concatenate(ann, batch_categories=None)
            graph_dict = SEDR.combine_graph_dict(graph_dict, graph_dict_tmp)
            name = name + '_' + sample
            
    del ann_list
    gc.collect()
    
    # Preprocess data
    # sc.pp.filter_genes(adata, min_cells=50)
    # sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=min(3000, adata.shape[1],adata.shape[0]))
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)
    # Perform PCA
    adata_X = PCA(n_components=min(pcs, adata.shape[1]-1,adata.shape[0]-1)).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X
    # Perform SEDR
    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device='cuda:0')
    using_dec = False
    if using_dec:
        sedr_net.train_with_dec()
    else:
        sedr_net.train_without_dec()
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['SEDR'] = sedr_feat

    meta_data = adata.obs[['batch']]
    data_mat = adata.obsm['SEDR']
    vars_use = ['batch']
    ho = hm.run_harmony(data_mat, meta_data, vars_use)
    res = pd.DataFrame(ho.Z_corr).T
    res_df = pd.DataFrame(data=res.values, columns=['X{}'.format(i+1) for i in range(res.shape[1])], index=adata.obs.index)
    adata.obsm['integrated'] = res_df.values
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata