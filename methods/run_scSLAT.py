import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np
import pandas as pd
import os
import cv2
from itertools import chain

from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match, run_SLAT_multi
from scSLAT.model.prematch import icp

def calculate_alpha(points: np.ndarray, method: str = "mean_distance", factor: float = 1.5) -> float:
    """
    Calculate an appropriate alpha value based on point distribution.

    Parameters:
    - points: np.ndarray, the input points.
    - method: str, the method to calculate alpha:
        - "mean_distance": Use the mean of nearest neighbor distances.
        - "median_distance": Use the median of nearest neighbor distances.
        - "std_distance": Use the standard deviation of distances.
    - factor: float, a multiplier to scale the computed alpha value.

    Returns:
    - alpha: float, the calculated alpha value.
    """
    nbrs = NearestNeighbors(n_neighbors=2).fit(points)
    distances, _ = nbrs.kneighbors(points)

    if method == "mean_distance":
        alpha = np.mean(distances[:, 1]) * factor
    elif method == "median_distance":
        alpha = np.median(distances[:, 1]) * factor
    elif method == "std_distance":
        alpha = np.std(distances[:, 1]) * factor
    else:
        raise ValueError("Invalid method. Choose 'mean_distance', 'median_distance', or 'std_distance'.")
    return alpha
    
def alpha_shape(points: np.ndarray, only_outer=True) -> tuple:
    """
    https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points
    """
    assert points.shape[0] > 3, "Need at least four points"

    alpha = calculate_alpha(points)
    print('alpha:', alpha)
    
    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points, if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    circum_r_list = []
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        circum_r_list.append(circum_r)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    boundary = list(set(list(chain.from_iterable(list(edges)))))  # noqa
    return boundary, edges, circum_r_list

def prematch(adata1, adata2):
    boundary_1, edges_1, _ = alpha_shape(adata1.obsm['spatial_aligned'], only_outer=True)
    boundary_2, edges_2, _ = alpha_shape(adata2.obsm['spatial_aligned'], only_outer=True)
    
    T, error = icp(adata2.obsm['spatial_aligned'][boundary_2,:].T, adata1.obsm['spatial_aligned'][boundary_1,:].T)
    rotation = np.arcsin(T[0,1]) * 360 / 2 / np.pi

    trans = np.squeeze(cv2.transform(np.array([adata2.obsm['spatial_aligned']], copy=True).astype(np.float32), T))[:,:2]
    adata2.obsm['spatial_aligned'] = trans
    adata2.obs['angle'] = rotation
    return adata2

#################### Run SLAT ####################
def run_scSLAT(ann_list, peak_data):
    for ann in ann_list:
        Cal_Spatial_Net(ann, k_cutoff=10, model='KNN')
    
    #### Stack slices
    for i in range(len(ann_list)):
        ann_list[i].obsm['spatial_aligned'] = ann_list[i].obsm['spatial']
    for i in range(len(ann_list)-1):
        ann_list[i+1] = prematch(ann_list[i], ann_list[i+1])
    adata = sc.concat(ann_list, index_unique=None)
    ### Perform SLAT
    if len(ann_list) == 2:
        if peak_data:
            edges, features = load_anndatas([ann_list[0], ann_list[1]], feature='glue')
        else:
            edges, features = load_anndatas([ann_list[0], ann_list[1]], feature='DPCA',check_order=False)
        embd0, embd1, time_tmp = run_SLAT(features, edges)
        best, index, distance = spatial_match([embd0, embd1], adatas=[ann_list[0],ann_list[1]], reorder=False)
        matching = np.array([range(index.shape[0]), best])
        matching_cell_ids = pd.DataFrame({
            'cell_id_1': ann_list[0].obs_names[matching[1]],
            'cell_id_2': ann_list[1].obs_names[matching[0]]     
        })
        rows = matching_cell_ids['cell_id_1'].unique()
        cols = matching_cell_ids['cell_id_2'].unique()
        results = pd.DataFrame(0, index=rows, columns=cols)
        for _, row in matching_cell_ids.iterrows():
            results.loc[row['cell_id_1'], row['cell_id_2']] = 1

        pi_list = [results]
        matching_cell_ids_list = [matching_cell_ids]
    else:
        matching_list, zip_res = run_SLAT_multi(ann_list, k_cutoff=10)
        
        pi_list = []
        matching_cell_ids_list = []
        for i in range(len(matching_list)):
            matching_cell_ids = pd.DataFrame({
                'cell_id_1': ann_list[i].obs_names[matching_list[i][1]],
                'cell_id_2': ann_list[i+1].obs_names[matching_list[i][0]]     
            })
            rows = matching_cell_ids['cell_id_1'].unique()
            cols = matching_cell_ids['cell_id_2'].unique()
            results = pd.DataFrame(0, index=rows, columns=cols)
            for _, row in matching_cell_ids.iterrows():
                results.loc[row['cell_id_1'], row['cell_id_2']] = 1
                
            pi_list.append(results)
            matching_cell_ids_list.append(matching_cell_ids)    
    return adata, pi_list, matching_cell_ids_list