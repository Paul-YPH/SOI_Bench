import warnings
warnings.filterwarnings("ignore")

import os

import numpy as np
import scanpy as sc

import STitch3D
from STitch3D.align_tools import icp, transform

def align_spots(adata_st_list_input, # list of spatial transcriptomics datasets
                data_type="visium", # a spot has six nearest neighborhoods if "Visium", four nearest neighborhoods otherwise
                coor_key="spatial", # "spatial" for visium; key for the spatial coordinates used for alignment
                tol=0.01, # parameter for "icp" method; tolerance level
                test_all_angles=False, # parameter for "icp" method; whether to test multiple rotation angles or not
                ):
    # Align coordinates of spatial transcriptomics

    # The first adata in the list is used as a reference for alignment
    adata_st_list = adata_st_list_input.copy()

    print("Using the Iterative Closest Point algorithm for alignemnt.")
    # Detect edges
    print("Detecting edges...")
    point_cloud_list = []
    for idx, adata in enumerate(adata_st_list):
        # Use in-tissue spots only
        # if 'in_tissue' in adata.obs.columns:
        #     adata = adata[adata.obs['in_tissue'] == 1]
        if isinstance(data_type, str) and data_type.lower() == "visium":
            loc_x = adata.obs.loc[:, ["array_row"]]
            loc_x = np.array(loc_x) * np.sqrt(3)
            loc_y = adata.obs.loc[:, ["array_col"]]
            loc_y = np.array(loc_y)
            loc = np.concatenate((loc_x, loc_y), axis=1)
            pairwise_loc_distsq = np.sum((loc.reshape([1,-1,2]) - loc.reshape([-1,1,2])) ** 2, axis=2)
            n_neighbors = np.sum(pairwise_loc_distsq < 5, axis=1) - 1
            edge = ((n_neighbors > 1) & (n_neighbors < 5)).astype(np.float32)
        else:
            loc_x = adata.obs.loc[:, ["array_row"]]
            loc_x = np.array(loc_x)
            loc_y = adata.obs.loc[:, ["array_col"]]
            loc_y = np.array(loc_y)
            loc = np.concatenate((loc_x, loc_y), axis=1)
            pairwise_loc_distsq = np.sum((loc.reshape([1,-1,2]) - loc.reshape([-1,1,2])) ** 2, axis=2)
            
            min_distsq = np.sort(np.unique(pairwise_loc_distsq), axis=None)[1]
            n_neighbors = np.sum(pairwise_loc_distsq < (min_distsq * 3), axis=1) - 1
            edge = ((n_neighbors > 1) & (n_neighbors < 7)).astype(np.float32)
                
        edge_points = np.sum(edge == 1)
        print(f"  Edge points: {edge_points}")
        if edge_points < 3:
            print(f"  Warning: Less than 3 points detected. Using all points instead.")
            point_cloud = adata.obsm[coor_key].copy()
            point_cloud_list.append(point_cloud)
        else:
            point_cloud = adata.obsm[coor_key][edge == 1].copy()
            point_cloud_list.append(point_cloud)    

    # Align edges
    print("Aligning edges...")
    trans_list = []
    rotation_angles = []  # List to store rotation angles
    adata_st_list[0].obsm["spatial_aligned"] = adata_st_list[0].obsm[coor_key].copy()
    # Calculate pairwise transformation matrices
    for i in range(len(adata_st_list) - 1):
        if test_all_angles == True:
            for angle in [0., np.pi * 1 / 3, np.pi * 2 / 3, np.pi, np.pi * 4 / 3, np.pi * 5 / 3]:
                R = np.array([[np.cos(angle), np.sin(angle), 0], 
                                [-np.sin(angle), np.cos(angle), 0], 
                                [0, 0, 1]]).T
                T, distances, _ = icp(transform(point_cloud_list[i+1], R), point_cloud_list[i], tolerance=tol)
                if angle == 0:
                    loss_best = np.mean(distances)
                    angle_best = angle
                    R_best = R
                    T_best = T
                else:
                    if np.mean(distances) < loss_best:
                        loss_best = np.mean(distances)
                        angle_best = angle
                        R_best = R
                        T_best = T
            T = T_best @ R_best
        else:
            T, _, _ = icp(point_cloud_list[i+1], point_cloud_list[i], tolerance=tol)
            
            R_best = T[:2, :2]  # Extract rotation matrix
            angle_best = np.arctan2(R_best[1, 0], R_best[0, 0]) # Calculate rotation angle
            
        trans_list.append(T)
        rotation_angles.append(np.degrees(angle_best))
        
    # Tranform
    for i in range(len(adata_st_list) - 1):
        point_cloud_align = adata_st_list[i+1].obsm[coor_key].copy()
        for T in trans_list[:(i+1)][::-1]:
            point_cloud_align = transform(point_cloud_align, T)
        adata_st_list[i+1].obsm["spatial_aligned"] = point_cloud_align
    return adata_st_list, rotation_angles

#################### Run STitch3D ####################
def run_STitch3D(ann_list):
    data_type = None
    if all(ann.obs['technology'].unique() == 'visium' for ann in ann_list):
        data_type = 'visium'
    # https://github.com/YangLabHKUST/STitch3D/blob/8da29785f6637c496e6660b44c5184ea434a7ec0/STitch3D/utils.py#L17
    ann_list, angle_list = align_spots(ann_list, data_type=data_type,tol=0.01)
    adata = sc.concat(ann_list, index_unique=None)
    return adata
    #adata.obs['angle'] = angle_list[0]
