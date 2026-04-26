import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import scanpy as sc
from STalign import STalign
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed

def mean_nearest_neighbor_distance(adata, k_cutoff=10):
    coords = adata.obsm["spatial"]
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=k_cutoff+1)  
    mean_distance = np.mean(distances[:, 1:])  
    return mean_distance

def run_STalign_2slices(ann_list,knn=10):
    # Rotation
    coords_I = np.array(ann_list[0].obsm['spatial'])
    coords_J = np.array(ann_list[1].obsm['spatial'])
    centroid1 = np.mean(coords_I, axis=0)
    centroid2 = np.mean(coords_J, axis=0)
    centered1 = coords_I - centroid1
    centered2 = coords_J - centroid2
    H = np.zeros((2, 2))
    for i in range(min(len(coords_I), len(coords_J))):
        H += np.outer(centered2[i], centered1[i])  
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    theta = np.arctan2(R[1, 0], R[0, 0])
    translation = centroid1 - (R @ centroid2)
    transformed_coords = (R @ coords_J.T).T + translation
    ann_list[1].obsm['spatial'] = transformed_coords

    dx0 = mean_nearest_neighbor_distance(ann_list[0],k_cutoff=knn)
    dx1 = mean_nearest_neighbor_distance(ann_list[1],k_cutoff=knn)
    
    a0 = mean_nearest_neighbor_distance(ann_list[0],k_cutoff=knn)*3
    a1 = mean_nearest_neighbor_distance(ann_list[1],k_cutoff=knn)*3
    
    ann_list[0].obsm['spatial_aligned'] = ann_list[0].obsm['spatial'].copy()
    
    # Perform STAlign
    xI = np.array(ann_list[1].obsm['spatial'].T[0]).astype(float)
    yI = np.array(ann_list[1].obsm['spatial'].T[1]).astype(float)
    xJ = np.array(ann_list[0].obsm['spatial'].T[0]).astype(float)
    yJ = np.array(ann_list[0].obsm['spatial'].T[1]).astype(float)
    XI,YI,I,fig = STalign.rasterize(xI,yI,dx=dx0)
    XJ,YJ,J,fig = STalign.rasterize(xJ,yJ,dx=dx1)
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    params = {
                'niter': 5000,
                'device':device,
                'epV': 50,
                'a': min(a0,a1)
            }
    
    out = STalign.LDDMM([YI,XI],I,[YJ,XJ],J,**params)
    A = out['A'].detach().cpu()
    v = out['v'].detach().cpu()
    xv = []
    for e in out['xv']:
        xv.append(e.detach().cpu())
    tpointsI= STalign.transform_points_source_to_target(xv,v,A, np.stack([yI, xI], 1))
    if tpointsI.is_cuda:
        tpointsI = tpointsI.cpu()
    xI_LDDMM = tpointsI[:,1]
    yI_LDDMM = tpointsI[:,0]
    ann_list[1].obsm['spatial_aligned'] = np.column_stack([xI_LDDMM, yI_LDDMM])
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    return adata

def run_STalign(adata, **args_dict):
    seed = args_dict['seed']
    knn = args_dict['knn']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    for ann in ann_list:
        ann.obsm['spatial_tmp'] = ann.obsm['spatial'].copy()
        ann.obsm['spatial_aligned'] = ann.obsm['spatial'].copy()
    
    for ii in range(len(ann_list) - 1):
        aligned_ann = run_STalign_2slices(ann_list[ii:ii + 2],knn=knn)
        aligned_ann = aligned_ann[ann_list[ii + 1].obs_names]
        ann_list[ii + 1].obsm['spatial'] = aligned_ann.obsm['spatial_aligned']
        ann_list[ii + 1].obsm['spatial_aligned'] = aligned_ann.obsm['spatial_aligned']
        
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    del ann_list
    gc.collect()
    
    adata.obsm['spatial'] = adata.obsm['spatial_tmp'].copy()
    del adata.obsm['spatial_tmp']
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata