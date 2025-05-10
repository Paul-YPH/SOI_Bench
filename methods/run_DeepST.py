import warnings
warnings.filterwarnings("ignore")

import os
import sys

import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

import torch

current_dir = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(os.path.join(current_dir, 'deepst'))
from deepst import DeepST
benchmark_dir = os.path.dirname(current_dir)  
sys.path.append(benchmark_dir)
from metrics.utils import *

#################### Run DeepST ####################
def run_DeepST(ann_list, sample_list, output_path):
    for ann in ann_list:
        ann.obs['cell_id'] = ann.obs_names
        ann.layers['counts'] = ann.X
    deepen = DeepST.run(save_path = output_path, 
	task = "Integration",
	pre_epochs = 800, 
	epochs = 1000, 
	use_gpu = True,
	)
    ###### Generate an augmented list of multiple datasets
    graph_list = []
    for ann in ann_list:
        # ann = process_h5ad(ann, quality="hires")
        ann = deepen._get_augment(ann, adjacent_weight=0.3, n_components=min(ann.X.shape[1], 200),neighbour_k=4, spatial_k=30,use_morphological=False)
        graph_dict = deepen._get_graph(ann.obsm['spatial'], distType='BallTree', k=12,
                                    rad_cutoff=150)
        # ann = deepen._get_image_crop(ann, data_name=ann.obs["batch"].values[0])
        # ann = deepen._get_augment(ann, spatial_type="LinearRegress")
        # graph_dict = deepen._get_graph(ann.obsm["spatial"], distType = "KDTree")
        graph_list.append(graph_dict) 

    ######## Synthetic Datasets and Graphs
    adata, multiple_graph = deepen._get_multiple_adata(adata_list = ann_list, data_name_list = sample_list, graph_list = graph_list)
    adata.obs_names = adata.obs['cell_index']
    ###### Enhanced data preprocessing
    data = deepen._data_process(adata, pca_n_comps = min(adata.shape[1], 200))
    deepst_embed = deepen._fit(
            data = data,
            graph_dict = multiple_graph,
            domains = adata.obs["batch"].values,  ##### Input to Domain Adversarial Model
            n_domains = len(sample_list))
    adata.obsm["DeepST_embed"] = deepst_embed
    adata.obsm["integrated"] = adata.obsm["DeepST_embed"]
    # Clustering
    adata = deepen._get_cluster_data(adata, n_domains=adata.obs["Ground Truth"].nunique(), priori = True)
    adata.obs.index = adata.obs['cell_id']
    adata.X = adata.layers['counts']

    return adata