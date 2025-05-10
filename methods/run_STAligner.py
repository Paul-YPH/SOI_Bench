import warnings
warnings.filterwarnings("ignore")

import os
import sys

import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
import torch
import math

import STAligner

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
      angle: rotation angle in radians
    '''

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    angle = np.arctan2(R[1, 0], R[0, 0])

    return T, R, t, angle
def ICP_align(adata_concat, adata_target, adata_ref, slice_target, slice_ref, landmark_domain):
    ### find MNN pairs in the landmark domain with knn=1    
    adata_slice1 = adata_target[adata_target.obs['louvain'].isin(landmark_domain)]
    adata_slice2 = adata_ref[adata_ref.obs['louvain'].isin(landmark_domain)]
    
    batch_pair = adata_concat[adata_concat.obs['batch_name'].isin([slice_target, slice_ref]) & adata_concat.obs['louvain'].isin(landmark_domain)]
    mnn_dict = STAligner.mnn_utils.create_dictionary_mnn(batch_pair, use_rep='STAligner', batch_name='batch_name', k=1, iter_comb=None, verbose=0)
    adata_1 = batch_pair[batch_pair.obs['batch_name']==slice_target]
    adata_2 = batch_pair[batch_pair.obs['batch_name']==slice_ref]
    
    anchor_list = []
    positive_list = []
    for batch_pair_name in mnn_dict.keys(): 
        for anchor in mnn_dict[batch_pair_name].keys():
            positive_spot = mnn_dict[batch_pair_name][anchor][0]
            ### anchor should only in the ref slice, pos only in the target slice
            if anchor in adata_1.obs_names and positive_spot in adata_2.obs_names:                 
                anchor_list.append(anchor)
                positive_list.append(positive_spot)
                             
    batch_as_dict = dict(zip(list(adata_concat.obs_names), range(0, adata_concat.shape[0])))
    anchor_ind = list(map(lambda _: batch_as_dict[_], anchor_list))
    positive_ind = list(map(lambda _: batch_as_dict[_], positive_list))
    anchor_arr = adata_concat.obsm['STAligner'][anchor_ind, ]
    positive_arr = adata_concat.obsm['STAligner'][positive_ind, ]
    dist_list = [np.sqrt(np.sum(np.square(anchor_arr[ii, :] - positive_arr[ii, :]))) for ii in range(anchor_arr.shape[0])]
    
    
    if all(d == 0 for d in dist_list):
        print("All distances are zero. Using all points.")
        key_points_src = np.array(anchor_list)  # Use all anchor points
        key_points_dst = np.array(positive_list)  # Use all positive points
    else:
        # Filter points based on distance threshold
        key_points_src = np.array(anchor_list)[dist_list < np.percentile(dist_list, 50)]  # Remove remote outliers
        key_points_dst = np.array(positive_list)[dist_list < np.percentile(dist_list, 50)]
    
    coor_src = adata_slice1.obsm["spatial"] ## to_be_aligned
    coor_dst = adata_slice2.obsm["spatial"] ## reference_points

    ## index number
    MNN_ind_src = [list(adata_1.obs_names).index(key_points_src[ii]) for ii in range(len(key_points_src))]
    MNN_ind_dst = [list(adata_2.obs_names).index(key_points_dst[ii]) for ii in range(len(key_points_dst))]
    
    
    ####### ICP alignment
    init_pose = None
    max_iterations = 100
    tolerance = 0.001

    coor_used = coor_src ## Batch_list[1][Batch_list[1].obs['annotation']==2].obsm["spatial"]
    coor_all = adata_target.obsm["spatial"].copy()
    coor_used = np.concatenate([coor_used, np.expand_dims(np.ones(coor_used.shape[0]), axis=1)], axis=1).T    
    coor_all = np.concatenate([coor_all, np.expand_dims(np.ones(coor_all.shape[0]), axis=1)], axis=1).T    
    A = coor_src  ## to_be_aligned
    B = coor_dst  ## reference_points

    m = A.shape[1] # get number of dimensions

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0

    total_angle = 0 
    
    for ii in range(max_iterations + 1):
        p1 = src[:m, MNN_ind_src].T
        p2 = dst[:m, MNN_ind_dst].T
        T, R, t, angle = best_fit_transform(src[:m, MNN_ind_src].T, dst[:m, MNN_ind_dst].T)
        
        if angle is not None:
            total_angle += np.degrees(angle)  # Convert radians to degrees and accumulate

        distances = np.mean([math.sqrt(((p1[kk, 0] - p2[kk, 0]) ** 2) + ((p1[kk, 1] - p2[kk, 1]) ** 2))
                             for kk in range(len(p1))])
        # update the current source
        src = np.dot(T, src)
        coor_used = np.dot(T, coor_used)
        coor_all = np.dot(T, coor_all)
        
        # check error
        mean_error = np.mean(distances)
        # print(mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
   
    aligned_points = coor_used.T    # MNNs in the landmark_domain
    aligned_points_all = coor_all.T # all points in the slice
    
    return aligned_points_all[:,:2], total_angle

def spatial_registration(ann_list, adata_concat, sample_list):
    n_slices = len(sample_list)
    ref_idx = 0 
    iter_comb = [(i, ref_idx) for i in range(1, n_slices)] 
    landmark_domain = [str(n_slices//2)]  

    ann_list[ref_idx].obsm["spatial_aligned"] = ann_list[ref_idx].obsm["spatial"].copy()
# https://staligner.readthedocs.io/en/latest/Tutorial_3D_alignment.html
# iter_comb = [(0,6), (1,6), (2,6), (3,6), (4,6), (5,6)]
# landmark_domain = ['3']
# for comb in iter_comb:
#     print(comb)
#     i, j = comb[0], comb[1]
#     adata_target = Batch_list[i]
#     adata_ref = Batch_list[j]
#     slice_target = section_ids[i]
#     slice_ref = section_ids[j]

#     aligned_coor = STAligner.ICP_align(adata_concat, adata_target, adata_ref, slice_target, slice_ref, landmark_domain)
#     adata_target.obsm["spatial"] = aligned_coor
    for comb in iter_comb:
        i, j = comb[0], comb[1]  
        adata_target = ann_list[i]
        adata_ref = ann_list[j]
        slice_target = ann_list[i].obs['batch'].values[0]
        slice_ref = ann_list[j].obs['batch'].values[0] 

        aligned_coor, angle = ICP_align(adata_concat, adata_target, adata_ref, slice_target, slice_ref, landmark_domain) 
        adata_target.obsm["spatial_aligned"] = aligned_coor
        
        ann_list[i] = adata_target
        ann_list[j] = adata_ref
    return ann_list 

current_dir = os.path.dirname(os.path.abspath(__file__)) 
benchmark_dir = os.path.dirname(current_dir)  
sys.path.append(benchmark_dir)
from metrics.utils import *

#################### Run STAligner ####################
def run_STAligner(ann_list, sample_list,cluster_option):
    adj_list = []
    for ann in ann_list:
        ann.X = sp.csr_matrix(ann.X)
        
        from sklearn.neighbors import NearestNeighbors
        spatial_coords = ann.obsm['spatial']
        nbrs = NearestNeighbors(n_neighbors=2).fit(spatial_coords) 
        distances, indices = nbrs.kneighbors(spatial_coords)
        nearest_distances = distances[:, 1]
        average_distance = np.mean(nearest_distances)
        
        STAligner.Cal_Spatial_Net(ann, rad_cutoff=average_distance * 1.5)
        # Normalization
        ann.layers['counts'] = ann.X.copy()
        sc.pp.normalize_total(ann, target_sum=1e4)
        sc.pp.log1p(ann)
        if ann.shape[1]>3000:   
            sc.pp.highly_variable_genes(ann, flavor="seurat_v3", n_top_genes=3000)
        else:
            ann.var['highly_variable'] = True
        ann = ann[:, ann.var['highly_variable']]
        adj_list.append(ann.uns['adj'])
    adata = anndata.concat(ann_list, join='inner', index_unique=None)
    adata.obs['original_clusters'] = adata.obs['Ground Truth'].astype('category')
    adata.obs["batch_name"] = adata.obs["batch"].astype('category')
    adj_concat = np.asarray(adj_list[0].todense())
    # Perform STAligner
    for batch_id in range(1,len(sample_list)):
        adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
    adata.uns['edgeList'] = np.nonzero(adj_concat)
    
    # adata = STAligner.train_STAligner(adata, verbose=True, knn_neigh = 50, device='cuda:0')
    def generate_iter_comb(ann_list):
        n = len(ann_list) 
        iter_comb = [(i, i + 1) for i in range(n - 1)] 
        return iter_comb
    iter_comb = generate_iter_comb(ann_list)
    # https://staligner.readthedocs.io/en/latest/Tutorial_3D_alignment.html
    adata = STAligner.train_STAligner_subgraph(adata, verbose=True, knn_neigh = min(100, adata.shape[0]), n_epochs = 1000, iter_comb = iter_comb,Batch_list=ann_list, device='cuda:0')
    adata.obsm['integrated'] = adata.obsm['STAligner']
    # Alignment
    sc.pp.neighbors(adata, use_rep='STAligner')
    sc.tl.louvain(adata, key_added="louvain", resolution=0.5)
    for it in range(len(sample_list)):
        adata_tmp = ann_list[it]
        adata_tmp.obs['louvain'] = adata.obs['louvain'][adata.obs['batch'] == sample_list[it]].values
        ann_list[it] = adata_tmp
    ann_list = spatial_registration(ann_list, adata, sample_list)
    #spatial_coords = []
    # angles = []
    # for i, ann in enumerate(ann_list):
    #     spatial_coords.append(ann.obsm['spatial'])
    #     angle = ann.obs['angle'].unique()[0]
    #     angles.append(angle)
    # adata.obsm['spatial_aligned'] = np.vstack(spatial_coords)
    # adata.obs['angle'] = angles[-1]
    tmp = sc.concat(ann_list, join='inner', index_unique=None)
    tmp = tmp[adata.obs_names,adata.var_names]
    adata.X = tmp.layers['counts']
    adata.obsm['spatial_aligned'] =tmp[adata.obs_names].obsm['spatial_aligned']
    del tmp
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=cluster_option)
    del adata.uns['edgeList']
    return adata