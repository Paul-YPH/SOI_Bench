###############################################
################## Alignment ##################
###############################################

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

# def compute_aas(adata, 
#                 batch_key='batch',
#                 pi=None,
#                 matching_list=None,
#                 angle_true=0):
    
#     if 'angle' in adata.obs.columns:
#         angle_r = abs(np.unique(adata.obs['angle'].values)[0])
#     else:
#         batch_list = adata.obs[batch_key].unique()
#         adata1 = adata[adata.obs[batch_key] == batch_list[0]]
#         adata2 = adata[adata.obs[batch_key] == batch_list[1]]
#         if matching_list is not None:
#             matching_df = matching_list[0]
#             adata1 = adata1[matching_df['cell_id_1']]
#             adata2 = adata2[matching_df['cell_id_2']]
#             X = np.array(adata1.obsm['spatial'])
#             Y = np.array(adata2.obsm['spatial'])
#         else:
#             X = np.array(adata1.obsm['spatial'])
#             Y = np.array(adata2.obsm['spatial'])
#         if pi is None:
#             R, t = find_rigid_transform(X, Y, ground_truth=angle_true)
#             angle_r = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))
#         else:
#             pi_df = pi[0]
#             angle_r = rotation_angle(X, Y, np.array(pi_df), ground_truth=angle_true)
#         angle_r = 360 - abs(angle_r)
    
#     return angle_r

def compute_aas(adata, 
                batch_key='batch',
                angle_true=None,
                matching_ground_truth=None):
    batch_list = adata.obs[batch_key].unique()
    
    print(batch_list)
    print(len(matching_ground_truth))
    
    
    if matching_ground_truth is None or angle_true is None:
        return None
    else:
        results = []
        num_matches = len(matching_ground_truth)
        
        for i in range(num_matches):  
            adata1 = adata[adata.obs[batch_key] == batch_list[0]]  
            adata2 = adata[adata.obs[batch_key] == batch_list[i + 1]]  
            
            matching_df = matching_ground_truth[i]  
            
            matching_df = matching_df[matching_df['cell_id_1'].isin(adata1.obs_names)]
            matching_df = matching_df[matching_df['cell_id_2'].isin(adata2.obs_names)]
            
            adata1 = adata1[matching_df['cell_id_1']]
            adata2 = adata2[matching_df['cell_id_2']]
            
            X = np.array(adata1.obsm['spatial_aligned'])
            Y = np.array(adata2.obsm['spatial_aligned'])
            
            R, t = find_rigid_transform(X, Y)
            angle_sub = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))
            angle_sub = 360 - abs(angle_sub)
            
            results.append({'metric': 'aas', 'value': [angle_sub], 'group': f'{batch_list[0]}_{batch_list[i+1]}'})
        
        df = pd.DataFrame(results)
        return df

def find_rigid_transform(A, B, ground_truth=None):
    if ground_truth is None:
        rad = 0
    else:
        rad = np.deg2rad(ground_truth)
    A = rotate_via_numpy(A, rad)
    
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def rotate_via_numpy(xy: np.ndarray, radians: float) -> np.ndarray:
    """
    https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
    """
    print(f"Rotation {radians * 180 / np.pi} degree")
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, xy.T).T
    return np.array(m)

# def rotation_angle(
#     X, Y, pi, ground_truth: float = 0, output_angle: bool = True, output_matrix: bool = False
# ):
#     """
#     https://github.com/gao-lab/SLAT/blob/dfb2b01d6fd52d5eeb88b8f17a69be6daaacdad4/scSLAT/metrics.py#L293
#     """
#     assert X.shape[1] == 2 and Y.shape[1] == 2

#     rad = np.deg2rad(ground_truth)
#     X = rotate_via_numpy(X, rad)

#     tX = pi.sum(axis=1).dot(X)
#     tY = pi.sum(axis=0).dot(Y)
#     X = X - tX
#     Y = Y - tY
#     H = Y.T.dot(pi.T.dot(X))
#     U, S, Vt = np.linalg.svd(H)
#     R = Vt.T.dot(U.T)
#     Y = R.dot(Y.T).T
#     M = np.array([[0, -1], [1, 0]])
#     theta = np.arctan2(np.trace(M.dot(H)), np.trace(H))
#     theta = -np.degrees(theta)
#     delta = np.absolute(theta - ground_truth)
#     if output_angle and not output_matrix:
#         return delta
#     elif output_angle and output_matrix:
#         return delta, X, Y, R, tX, tY
#     else:
#         return X, Y
