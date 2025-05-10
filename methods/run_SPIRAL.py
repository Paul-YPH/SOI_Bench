import warnings
warnings.filterwarnings("ignore")

import os
import argparse

import numpy as np
import pandas as pd
import sklearn
import torch
import anndata
import scanpy as sc
import scipy

from spiral.main import SPIRAL_integration
from spiral.layers import *
from spiral.utils import *
from spiral.CoordAlignment import CoordAlignment


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
#     coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net
    
#################### Run SPIRAL ####################
def run_SPIRAL(ann_list, sample_list, output_path):
    # Process data
    print('### Processing data...')
    IDX = np.arange(0, len(sample_list))
    VF = []
    MAT = []
    flags = str(sample_list[IDX[0]])
    
    for i in np.arange(1, len(IDX)):
        flags = flags + '-' + str(sample_list[IDX[i]])
    flags = flags + "_"
    
    for k in np.arange(len(IDX)):
        selected_sample = str(sample_list[IDX[k]])
        ann = ann_list[k].copy()
        ann.var_names_make_unique()
        ann.layers['counts'] = ann.X

        sc.pp.normalize_total(ann, target_sum=1e4)
        sc.pp.log1p(ann)
        if ann.shape[1]>3000:
            sc.pp.highly_variable_genes(ann, flavor="seurat_v3", n_top_genes=3000)
        else:
            ann.var['highly_variable'] = True

        mat1 = pd.DataFrame(ann.X.toarray() if isinstance(ann.X, scipy.sparse.spmatrix) else ann.X, columns=ann.var_names, index=ann.obs_names)

        coord1 = pd.DataFrame(ann.obsm['spatial'], columns=['x','y'], index=ann.obs_names)
        meta1 = ann.obs[['Ground Truth', 'batch']]
        meta1.columns = ['celltype', 'batch']
        meta1.index = ann.obs_names
        
        gtt_input_scanpy_dir = os.path.join(output_path, "gtt_input_scanpy")
        if not os.path.exists(gtt_input_scanpy_dir):
            os.makedirs(gtt_input_scanpy_dir)
        
        meta1.to_csv(os.path.join(gtt_input_scanpy_dir, flags+str(selected_sample)+"_label-1.txt"))
        coord1.to_csv(os.path.join(gtt_input_scanpy_dir, flags+str(selected_sample)+"_positions-1.txt"))
        MAT.append(mat1)
        VF = np.union1d(VF, ann.var_names[ann.var['highly_variable']])
        
    for i in np.arange(len(IDX)):
        mat = MAT[i]
        mat = mat.loc[:,VF]
        mat.to_csv(os.path.join(gtt_input_scanpy_dir, flags+str(sample_list[IDX[i]])+"_features-1.txt"))
        
    rad=150
    KNN=6

    flags=str(sample_list[0])
    for i in range(1,len(sample_list)):
        flags=flags+'-'+str(sample_list[i])
    for i in range(len(sample_list)):
        sample1=sample_list[i]
        base_path = os.path.join(output_path, "gtt_input_scanpy", flags + '_' + str(sample1))
        features = pd.read_csv(base_path + "_features-1.txt", header=0, index_col=0, sep=',')
        meta = pd.read_csv(base_path + "_label-1.txt", header=0, index_col=0, sep=',') 
        coord = pd.read_csv(base_path + "_positions-1.txt", header=0, index_col=0, sep=',')
        # meta=meta.iloc[:meta.shape[0]-1,:]
        adata = sc.AnnData(features)
        adata.var_names_make_unique()
        adata.X=scipy.sparse.csr_matrix(adata.X)
        adata.obsm["spatial"] = coord.loc[:,['x','y']].to_numpy()
        Cal_Spatial_Net(adata, rad_cutoff=rad, k_cutoff=6, model='KNN', verbose=True)
        if 'highly_variable' in adata.var.columns:
            adata_Vars =  adata[:, adata.var['highly_variable']]
        else:
            adata_Vars = adata
        features = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)
        cells = np.array(features.index)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

        Spatial_Net = adata.uns['Spatial_Net']
        G_df = Spatial_Net.copy()
        np.savetxt(os.path.join(gtt_input_scanpy_dir, flags+'_'+str(sample1)+"_edge_KNN_"+str(KNN)+".csv"), G_df.values[:,:2], fmt='%s')


    samples=sample_list[0:sample_list.__len__()]
    SEP=','
    net_cate='_KNN_'

    knn=6
    N_WALKS=knn
    WALK_LEN=1
    N_WALK_LEN=knn
    NUM_NEG=knn

    feat_file=[]
    edge_file=[]
    meta_file=[]
    coord_file=[]
    flags=''
    flags1=str(samples[0])
    for i in range(1,len(samples)):
        flags1=flags1+'-'+str(samples[i])
    for i in range(len(samples)):
        feat_file.append(os.path.join(output_path, "gtt_input_scanpy", f"{flags1}_{samples[i]}_features-1.txt"))
        edge_file.append(os.path.join(output_path, "gtt_input_scanpy", f"{flags1}_{samples[i]}_edge_KNN_{knn}.csv"))
        meta_file.append(os.path.join(output_path, "gtt_input_scanpy", f"{flags1}_{samples[i]}_label-1.txt"))
        coord_file.append(os.path.join(output_path, "gtt_input_scanpy", f"{flags1}_{samples[i]}_positions-1.txt"))
        flags = flags + '_' + str(samples[i])
    N=pd.read_csv(feat_file[0],header=0,index_col=0).shape[1]
    if (len(samples)==2):
        M=1
    else:
        M=len(samples)
        


    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='The seed of initialization.')
    parser.add_argument('--AEdims', type=list, default=[N,[512],32], help='Dim of encoder.')
    parser.add_argument('--AEdimsR', type=list, default=[32,[512],N], help='Dim of decoder.')
    parser.add_argument('--GSdims', type=list, default=[512,32], help='Dim of GraphSAGE.')
    parser.add_argument('--zdim', type=int, default=32, help='Dim of embedding.')
    parser.add_argument('--znoise_dim', type=int, default=4, help='Dim of noise embedding.')
    parser.add_argument('--CLdims', type=list, default=[4,[],M], help='Dim of classifier.')
    parser.add_argument('--DIdims', type=list, default=[28,[32,16],M], help='Dim of discriminator.')
    parser.add_argument('--beta', type=float, default=1.0, help='weight of GraphSAGE.')
    parser.add_argument('--agg_class', type=str, default=MeanAggregator, help='Function of aggregator.')
    parser.add_argument('--num_samples', type=str, default=knn, help='number of neighbors to sample.')

    parser.add_argument('--N_WALKS', type=int, default=N_WALKS, help='number of walks of random work for postive pairs.')
    parser.add_argument('--WALK_LEN', type=int, default=WALK_LEN, help='walk length of random work for postive pairs.')
    parser.add_argument('--N_WALK_LEN', type=int, default=N_WALK_LEN, help='number of walks of random work for negative pairs.')
    parser.add_argument('--NUM_NEG', type=int, default=NUM_NEG, help='number of negative pairs.')


    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    
    min_cells = min(ann.shape[0] for ann in ann_list)
    parser.add_argument('--batch_size', type=int, default=min(min_cells, 1024), help='Size of batches to train.') ####512 for withon donor;1024 for across donor###
    
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    
    
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--alpha1', type=float, default=N, help='Weight of decoder loss.')
    parser.add_argument('--alpha2', type=float, default=1, help='Weight of GraphSAGE loss.')
    parser.add_argument('--alpha3', type=float, default=1, help='Weight of classifier loss.')
    parser.add_argument('--alpha4', type=float, default=1, help='Weight of discriminator loss.')
    parser.add_argument('--lamda', type=float, default=1, help='Weight of GRL.')
    parser.add_argument('--Q', type=float, default=10, help='Weight negative loss for sage losss.')

    params,unknown=parser.parse_known_args()

    SPII=SPIRAL_integration(params,feat_file,edge_file,meta_file)
    SPII.train()

    SPII.model.eval()
    all_idx=np.arange(SPII.feat.shape[0])
    all_layer,all_mapping=layer_map(all_idx.tolist(),SPII.adj,len(SPII.params.GSdims))
    all_rows=SPII.adj.tolil().rows[all_layer[0]]
    all_feature=torch.Tensor(SPII.feat.iloc[all_layer[0],:].values).float().cuda()
    all_embed,ae_out,clas_out,disc_out=SPII.model(all_feature,all_layer,all_mapping,all_rows,SPII.params.lamda,SPII.de_act,SPII.cl_act)
    [ae_embed,gs_embed,embed]=all_embed
    [x_bar,x]=ae_out
    embed=embed.cpu().detach()
    names=['GTT_'+str(i) for i in range(embed.shape[1])]
    embed1=pd.DataFrame(np.array(embed),index=SPII.feat.index,columns=names)

    gtt_output_dir = os.path.join(output_path, "gtt_output")
    if not os.path.exists(gtt_output_dir):
        os.makedirs(gtt_output_dir)
        
    embed_file=os.path.join(gtt_output_dir, "SPIRAL"+flags+"_embed_"+str(SPII.params.batch_size)+".csv")
    embed1.to_csv(embed_file)
    meta=SPII.meta.values

    embed_df = pd.DataFrame(embed.cpu().numpy(), index=SPII.feat.index, columns=names)
    embed_new = torch.cat((
        torch.zeros((embed_df.shape[0], SPII.params.znoise_dim)),
        torch.Tensor(embed_df.iloc[:, SPII.params.znoise_dim:].values)
    ), dim=1)

    xbar_new=np.array(SPII.model.agc.ae.de(embed_new.cuda(),nn.Sigmoid())[1].cpu().detach())
    xbar_new1=pd.DataFrame(xbar_new,index=SPII.feat.index,columns=SPII.feat.columns)

    xbar_new1.to_csv(os.path.join(gtt_output_dir, "SPIRAL"+flags+"_correct_"+str(SPII.params.batch_size)+".csv"))

    adata=anndata.AnnData(SPII.feat)
    adata.obsm['spiral']=embed1.iloc[:,SPII.params.znoise_dim:].values
    adata.obs['batch']=SPII.meta.loc[:,'batch'].values
    adata.obs['Ground Truth']=SPII.meta.loc[:,'celltype'].values
    ub=np.unique(adata.obs['batch'])
    sc.pp.neighbors(adata,use_rep='spiral')
    adata.obsm['integrated']=adata.obsm['spiral']

    n_clust=SPII.meta['celltype'].dropna().nunique()
    adata = mclust_R(adata, used_obsm='spiral', num_cluster=n_clust)
    cluster_file=os.path.join(gtt_output_dir, "SPIRAL"+flags+"_mclust.csv")
    pd.DataFrame(adata.obs['mclust']).to_csv(cluster_file)

    coord=pd.read_csv(coord_file[0],header=0,index_col=0)
    for i in np.arange(1,len(samples)):
        coord=pd.concat((coord,pd.read_csv(coord_file[i],header=0,index_col=0)))
    coord.columns=['y','x']
    # coord=pd.DataFrame()
    # for i in range(len(ann_list)):
    #     tmp=pd.DataFrame(ann_list[i].obsm['spatial'],index=ann_list[i].obs_names  )
    #     coord=pd.concat((coord,tmp))
    # adata.obsm['spatial']=coord.loc[adata.obs_names,:].values

    clust_cate='louvain'
    input_file=[meta_file,coord_file,embed_file,cluster_file]
    output_dirs=os.path.join(output_path, "gtt_output/SPIRAL_alignment/")
    if not os.path.exists(output_dirs):
        os.makedirs(output_dirs)

    alpha=0.5
    types="weighted_mean"
    CA=CoordAlignment(input_file=input_file,output_dirs=output_dirs,ub=ub,flags=flags,
                    clust_cate=clust_cate,R_dirs='/usr/lib/R',alpha=alpha,types=types)
    New_Coord=CA.New_Coord
    New_Coord = New_Coord.reindex(adata.obs.index)
    adata.obsm['spatial_aligned']=New_Coord[['x','y']].values
    adata.obs['benchmark_cluster']=New_Coord['clusters']
    
    all_spatial = []
    all_obs = []
    for ann in ann_list:
        all_spatial.append(pd.DataFrame(ann.obsm['spatial'], index=ann.obs_names))
    spatial_combined = pd.concat(all_spatial)
    adata.obsm['spatial'] = spatial_combined.loc[adata.obs_names].values
    
    tmp = sc.concat(ann_list, join='outer', index_unique=None)
    tmp = tmp[adata.obs_names,adata.var_names]
    adata.X = tmp.X
    del tmp
    return adata