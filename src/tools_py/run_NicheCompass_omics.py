import warnings
warnings.filterwarnings("ignore")

import os
import random
import warnings
from datetime import datetime

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
import gc
from nichecompass.models import NicheCompass
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                add_multimodal_mask_to_adata,
                                create_new_color_dict,
                                compute_communication_gp_network,
                                visualize_communication_gp_network,
                                extract_gp_dict_from_collectri_tf_network,
                                extract_gp_dict_from_mebocost_ms_interactions,
                                extract_gp_dict_from_nichenet_lrt_interactions,
                                extract_gp_dict_from_omnipath_lr_interactions,
                                filter_and_combine_gp_dict_gps_v2,
                                get_gene_annotations,
                                generate_enriched_gp_info_plots,
                                generate_multimodal_mapping_dict,
                                get_unique_genes_from_gp_dict)
from utils import get_ann_list, create_lightweight_adata, set_seed
from clustering import clustering

def run_NicheCompass_omics(adata, path, **args_dict):
    seed = args_dict['seed']
    clust = args_dict['clust']
    knn = args_dict['knn']
    set_seed(seed)
    
    if clust == 'mclust':
        print('Change clustering method from mclust to leiden')
        clust = 'leiden'
    
    n_neighbors=knn
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    ### Dataset ###
    adata_omics1 = ann_list[config['sample_list'].tolist().index('rna')]
    adata_omics2 = ann_list[config['sample_list'].tolist().index('atac')]

    adata_omics1.obs_names = [n.split('_', 1)[1] for n in adata_omics1.obs_names]
    adata_omics2.obs_names = [n.split('_', 1)[1] for n in adata_omics2.obs_names]
    
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    adata_omics1.X = adata_omics1.X.astype('float32')
    adata_omics2.X = adata_omics2.X.astype('float32')
    adata_omics1.layers['count'] = adata_omics1.X.copy()  
    adata_omics2.layers['count'] = adata_omics2.X.copy()  
    species = adata_omics1.obs['species'].unique()[0].lower()
    
    if species not in ['mouse', 'human']:
        print(f"Species {species} not supported, using mouse as default.")
        species = 'mouse'
    
    def format_var_name(var_name):
        parts = var_name.split("-")
        if len(parts) == 3:
            return f"{parts[0]}:{parts[1]}-{parts[2]}"
        return var_name  

    adata_omics2.var.index = adata_omics2.var.index.to_series().apply(format_var_name)
    adata_omics2.var.names = adata_omics2.var.index

    ### Dataset ###
    spatial_key = "spatial"
    n_sampled_neighbors = 4
    filter_genes = True
    n_svg = 3000
    n_svp = 15000
    filter_peaks = True
    min_cell_peak_thresh_ratio = 0.005 # 0.05%
    min_cell_gene_thresh_ratio = 0.005 # 0.05%
    
    ### Model ###
    # AnnData keys
    counts_key = "count"
    adj_key = "spatial_connectivities"
    gp_names_key = "nichecompass_gp_names"
    active_gp_names_key = "nichecompass_active_gp_names"
    gp_targets_mask_key = "nichecompass_gp_targets"
    gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
    gp_sources_mask_key = "nichecompass_gp_sources"
    gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
    latent_key = "nichecompass_latent"
    
    # Architecture
    active_gp_thresh_ratio = 0.01
    conv_layer_encoder = "gcnconv" # change to "gcnconv" if not enough compute and memory default:gatconv

    # Trainer
    n_epochs = 400
    n_epochs_all_gps = 25
    lr = 0.0001
    lambda_edge_recon = 500000.
    lambda_gene_expr_recon = 300.
    lambda_chrom_access_recon = 300.
    lambda_l1_masked = 0. # prior GP  regularization
    lambda_l1_addon = 30. # de novo GP regularization
    edge_batch_size = 128 # increase if more memory available or decrease to save memory default:256
    use_cuda_if_available = True

    ### Analysis ###
    cell_type_key = "cell_type"
    latent_leiden_resolution = 0.6
    latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
    sample_key = "batch"
    spot_size = 30
    differential_gp_test_results_key = "nichecompass_differential_gp_test_results"
    
    now = datetime.now()
    current_timestamp = now.strftime("%d%m%Y_%H%M%S")
    
    # Define paths
    ga_data_folder_path = f"{path}/gene_annotations"
    gp_data_folder_path = f"{path}/gene_programs"
    so_data_folder_path = f"{path}/spatial_omics"
    artifacts_folder_path = f"{path}/artifacts"
    if not os.path.exists(so_data_folder_path):
        os.makedirs(so_data_folder_path, exist_ok=True)
    if not os.path.exists(gp_data_folder_path):
        os.makedirs(gp_data_folder_path, exist_ok=True)
    if not os.path.exists(ga_data_folder_path):
        os.makedirs(ga_data_folder_path, exist_ok=True)
    if not os.path.exists(artifacts_folder_path):
        os.makedirs(artifacts_folder_path, exist_ok=True)

    omnipath_lr_network_file_path = f"{gp_data_folder_path}/omnipath_lr_network.csv"
    collectri_tf_network_file_path = f"{gp_data_folder_path}/collectri_tf_network_{species}.csv"
    nichenet_lr_network_file_path = f"{gp_data_folder_path}/nichenet_lr_network_v2_{species}.csv"
    nichenet_ligand_target_matrix_file_path = f"{gp_data_folder_path}/nichenet_ligand_target_matrix_v2_{species}.csv"
    mebocost_enzyme_sensor_interactions_folder_path = f"{gp_data_folder_path}/metabolite_enzyme_sensor_gps"
    gene_orthologs_mapping_file_path = f"{ga_data_folder_path}/human_mouse_gene_orthologs.csv"
    gtf_file_path = f"{ga_data_folder_path}/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz"

    model_folder_path = f"{artifacts_folder_path}/sample_integration/{current_timestamp}/model"
    figure_folder_path = f"{artifacts_folder_path}/sample_integration/{current_timestamp}/figures"

    os.makedirs(model_folder_path, exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)
    os.makedirs(so_data_folder_path, exist_ok=True)
    
    # Retrieve OmniPath GPs (source: ligand genes; target: receptor genes)
    omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
        species=species,
        load_from_disk=False,
        save_to_disk=False,
        lr_network_file_path=omnipath_lr_network_file_path,
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        plot_gp_gene_count_distributions=False)
    
    # Display example OmniPath GP
    omnipath_gp_names = list(omnipath_gp_dict.keys())
    random.shuffle(omnipath_gp_names)
    omnipath_gp_name = omnipath_gp_names[0]
    print(f"{omnipath_gp_name}: {omnipath_gp_dict[omnipath_gp_name]}")
    
    # Retrieve NicheNet GPs (source: ligand genes; target: receptor genes, target genes)
    nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
        species=species,
        version="v2",
        keep_target_genes_ratio=1.,
        max_n_target_genes_per_gp=250,
        load_from_disk=False,
        save_to_disk=False,
        lr_network_file_path=nichenet_lr_network_file_path,
        ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        plot_gp_gene_count_distributions=False)
    
    # Retrieve MEBOCOST GPs (source: enzyme genes; target: sensor genes)
    mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
        dir_path=mebocost_enzyme_sensor_interactions_folder_path,
        species=species,
        plot_gp_gene_count_distributions=False)

    # Retrieve CollecTRI GPs (source: -; target: transcription factor genes, target genes)
    collectri_gp_dict = extract_gp_dict_from_collectri_tf_network(
            species=species,
            tf_network_file_path=collectri_tf_network_file_path,
            load_from_disk=False,
            save_to_disk=False,
            plot_gp_gene_count_distributions=False)
    
    # Filter and combine GPs
    gp_dicts = [omnipath_gp_dict, nichenet_gp_dict, mebocost_gp_dict, collectri_gp_dict]
    combined_gp_dict = filter_and_combine_gp_dict_gps_v2(
        gp_dicts,
        verbose=True)

    print(f"Number of gene programs after filtering and combining: "
        f"{len(combined_gp_dict)}.")
    
    # Compute spatial neighborhood
    sq.gr.spatial_neighbors(adata_omics1,
                            coord_type="generic",
                            spatial_key=spatial_key,
                            n_neighs=n_neighbors)

    # Make adjacency matrix symmetric
    adata_omics1.obsp[adj_key] = (
        adata_omics1.obsp[adj_key].maximum(
            adata_omics1.obsp[adj_key].T))
    
    if filter_genes:
        print("Filtering genes...")
        # Filter genes and only keep ligand, receptor, enzyme, sensor, and
        # the 'n_svg' spatially variable genes
        gp_dict_genes = get_unique_genes_from_gp_dict(
            gp_dict=combined_gp_dict,
                retrieved_gene_entities=["sources", "targets"])
        print(f"Starting with {len(adata_omics1.var_names)} genes.")
        min_cells = int(adata_omics1.shape[0] * min_cell_gene_thresh_ratio)
        sc.pp.filter_genes(adata_omics1, min_cells=min_cells)
        print(f"Keeping {len(adata_omics1.var_names)} genes after filtering genes with "
            f"counts in less than {int(adata_omics1.shape[0] * min_cell_gene_thresh_ratio)} cells.")
        
        # Identify spatially variable genes
        sq.gr.spatial_autocorr(adata_omics1, mode="moran", genes=adata_omics1.var_names)
        svg_genes = adata_omics1.uns["moranI"].index[:n_svg].tolist()
        adata_omics1.var["spatially_variable"] = adata_omics1.var_names.isin(svg_genes)
        adata_omics1 = adata_omics1[:, adata_omics1.var["spatially_variable"] == True]
        print(f"Keeping {len(adata_omics1.var_names)} spatially variable genes.")
        
    if filter_peaks:
        print("\nFiltering peaks...")
        print(f"Starting with {len(adata_omics2.var_names)} peaks.")
        # Filter out peaks that are rarely detected to reduce GPU footprint of model
        min_cells = int(adata_omics2.shape[0] * min_cell_peak_thresh_ratio)
        sc.pp.filter_genes(adata_omics2, min_cells=min_cells)
        print(f"Keeping {len(adata_omics2.var_names)} peaks after filtering peaks with "
            f"counts in less than {int(adata_omics2.shape[0] * min_cell_peak_thresh_ratio)} cells.")
        
        # Filter spatially variable peaks
        adata_omics2.obsp["spatial_connectivities"] = adata_omics1.obsp["spatial_connectivities"]
        adata_omics2.obsp["spatial_distances"] = adata_omics1.obsp["spatial_distances"]

        sq.gr.spatial_autocorr(adata_omics2,
                            mode="moran",
                            genes=adata_omics2.var_names)
        sv_peaks = adata_omics2.uns["moranI"].index[:n_svp].tolist()
        adata_omics2.var["spatially_variable"] = adata_omics2.var_names.isin(sv_peaks)
        adata_omics2 = adata_omics2[:, adata_omics2.var["spatially_variable"] == True]
        print(f"Keeping {len(adata_omics2.var_names)} peaks after filtering spatially variable "
            f"peaks.")
        
    adata_omics1, adata_omics2 = get_gene_annotations(
        adata=adata_omics1,
        adata_atac=adata_omics2,
        gtf_file_path=gtf_file_path)

    adata_omics1.var = adata_omics1.var.loc[:, ~adata_omics1.var.columns.duplicated()]
    valid_vars = adata_omics1.var.index[adata_omics1.var["chrom"].notnull()]
    adata_omics1 = adata_omics1[:, valid_vars]
    
    # Add the GP dictionary as binary masks to the adata
    add_gps_from_gp_dict_to_adata(
        gp_dict=combined_gp_dict,
        adata=adata_omics1,
        gp_targets_mask_key=gp_targets_mask_key,
        gp_targets_categories_mask_key=gp_targets_categories_mask_key,
        gp_sources_mask_key=gp_sources_mask_key,
        gp_sources_categories_mask_key=gp_sources_categories_mask_key,
        gp_names_key=gp_names_key,
        min_genes_per_gp=2,
        min_source_genes_per_gp=0,
        min_target_genes_per_gp=1,
        max_genes_per_gp=None,
        max_source_genes_per_gp=None,
        max_target_genes_per_gp=None,
        plot_gp_gene_count_distributions=False)
    
    gene_peak_mapping_dict = generate_multimodal_mapping_dict(
        adata=adata_omics1,
        adata_atac=adata_omics2)
    
    adata_omics1, adata_omics2 = add_multimodal_mask_to_adata(
        adata=adata_omics1,
        adata_atac=adata_omics2,
        gene_peak_mapping_dict=gene_peak_mapping_dict)

    print(f"Keeping {len(adata_omics2.var_names)} peaks after filtering peaks with "
        "no matching genes in gp mask.")
    

    # Initialize model
    model = NicheCompass(adata_omics1,
                        adata_omics2,
                        counts_key=counts_key,
                        adj_key=adj_key,
                        gp_names_key=gp_names_key,
                        active_gp_names_key=active_gp_names_key,
                        gp_targets_mask_key=gp_targets_mask_key,
                        gp_targets_categories_mask_key=gp_targets_categories_mask_key,
                        gp_sources_mask_key=gp_sources_mask_key,
                        gp_sources_categories_mask_key=gp_sources_categories_mask_key,
                        latent_key=latent_key,
                        conv_layer_encoder=conv_layer_encoder,
                        active_gp_thresh_ratio=active_gp_thresh_ratio)
        
    # Train model
    model.train(n_epochs=n_epochs,
                n_epochs_all_gps=n_epochs_all_gps,
                lr=lr,
                lambda_edge_recon=lambda_edge_recon,
                lambda_gene_expr_recon=lambda_gene_expr_recon,
                lambda_chrom_access_recon=lambda_chrom_access_recon,
                lambda_l1_masked=lambda_l1_masked,
                lambda_l1_addon=lambda_l1_addon,
                edge_batch_size=edge_batch_size,
                use_cuda_if_available=use_cuda_if_available,
                n_sampled_neighbors=n_sampled_neighbors,
                verbose=True)
    
    adata = model.adata
    adata.obsm['integrated'] = adata.obsm['nichecompass_latent']
    
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata