###############################################
################## Alignment ##################
###############################################

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

def compute_aas(adata, 
                batch_key='batch'):
    
    batch_list = adata.obs[batch_key].unique()
    results = []
    ref_batch = batch_list[0]
    
    for i in range(1, len(batch_list)):
        query_batch = batch_list[i]
        
        adata1 = adata[adata.obs[batch_key] == ref_batch].copy()
        adata2 = adata[adata.obs[batch_key] == batch_list[i]].copy()
        
        prefix1 = f"{ref_batch}_"
        prefix2 = f"{query_batch}_"
        
        adata1.obs['temp_orig_id'] = adata1.obs_names.str.replace(prefix1, '', regex=False)
        adata2.obs['temp_orig_id'] = adata2.obs_names.str.replace(prefix2, '', regex=False)
        
        common_ids = np.intersect1d(adata1.obs['temp_orig_id'], adata2.obs['temp_orig_id'])
        
        if len(common_ids) == 0:
            print(f"Warning: No matching cells found between {ref_batch} and {query_batch}")
            continue
            
        adata1 = adata1[adata1.obs['temp_orig_id'].isin(common_ids)]
        adata2 = adata2[adata2.obs['temp_orig_id'].isin(common_ids)]
        
        adata1 = adata1[adata1.obs['temp_orig_id'].argsort()]
        adata1_map = dict(zip(adata1.obs['temp_orig_id'], adata1.obs_names))
        adata2_map = dict(zip(adata2.obs['temp_orig_id'], adata2.obs_names))
        
        sorted_common_ids = sorted(common_ids)
        obs_names_1 = [adata1_map[uid] for uid in sorted_common_ids]
        obs_names_2 = [adata2_map[uid] for uid in sorted_common_ids]
        
        adata1 = adata1[obs_names_1]
        adata2 = adata2[obs_names_2]
        
        X = np.array(adata1.obsm['spatial_aligned'])
        Y = np.array(adata2.obsm['spatial_aligned'])
        
        R, t = find_rigid_transform(X, Y)
        angle_sub = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))
        angle_sub = 360 - abs(angle_sub)
        
        results.append({'metric': 'aas', 'value': angle_sub, 'group': f'{ref_batch}_{query_batch}'})
    
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