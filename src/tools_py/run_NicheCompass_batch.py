import warnings
warnings.filterwarnings("ignore")

import os
import random
import warnings
from datetime import datetime

import anndata as ad
import scipy.sparse as sp
import squidpy as sq
import scanpy as sc
import gc
from nichecompass.models import NicheCompass
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                extract_gp_dict_from_mebocost_ms_interactions,
                                extract_gp_dict_from_nichenet_lrt_interactions,
                                extract_gp_dict_from_omnipath_lr_interactions,
                                filter_and_combine_gp_dict_gps_v2
                                )
from utils import get_ann_list, create_lightweight_adata, set_seed
from clustering import clustering

def run_NicheCompass_batch(adata, path, **args_dict):
    clust = args_dict['clust']
    seed = args_dict['seed']
    knn = args_dict['knn']
    set_seed(seed)
    
    if clust == 'mclust':
        print('Change clustering method from mclust to leiden')
        clust = 'leiden'
    
    n_neighbors=knn
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    for ann in ann_list:
        sc.pp.filter_cells(ann, min_genes=1)
        sc.pp.filter_genes(ann, min_cells=1)
        ann.X = ann.X.astype('float32')
        ann.layers['count'] = ann.X.copy()

    ### Dataset ###
    species = ann_list[0].obs['species'].unique()[0].lower()
    if species not in ['mouse', 'human']:
        print(f"Species {species} not supported, using mouse as default.")
        species = 'mouse'
    spatial_key = "spatial"

    ### Model ###
    # AnnData keys
    counts_key = "count"
    adj_key = "spatial_connectivities"
    cat_covariates_keys = ["batch"]
    gp_names_key = "nichecompass_gp_names"
    active_gp_names_key = "nichecompass_active_gp_names"
    gp_targets_mask_key = "nichecompass_gp_targets"
    gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
    gp_sources_mask_key = "nichecompass_gp_sources"
    gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
    latent_key = "nichecompass_latent"

    # Architecture
    cat_covariates_embeds_injection = ["gene_expr_decoder"]
    cat_covariates_embeds_nums = [3]
    cat_covariates_no_edges = [True]
    conv_layer_encoder = "gcnconv" # change to "gatv2conv" if enough compute and memory
    active_gp_thresh_ratio = 0.01

    # Trainer
    n_epochs = 400
    n_epochs_all_gps = 25
    lr = 0.001
    # lr = 0.0001 #D46,47
    lambda_edge_recon = 500000.
    lambda_gene_expr_recon = 300.
    lambda_l1_masked = 0. # prior GP  regularization
    lambda_l1_addon = 30. # de novo GP regularization
    edge_batch_size = 4096 # increase if more memory available or decrease to save memory
    n_sampled_neighbors = 4
    use_cuda_if_available = True

    ### Analysis ###
    # cell_type_key = "Main_molecular_cell_type"
    # latent_leiden_resolution = 0.2
    # latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
    # sample_key = "batch"
    # spot_size = 0.2
    # differential_gp_test_results_key = "nichecompass_differential_gp_test_results"

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

    model_folder_path = f"{artifacts_folder_path}/sample_integration/{current_timestamp}/model"
    figure_folder_path = f"{artifacts_folder_path}/sample_integration/{current_timestamp}/figures"

    os.makedirs(model_folder_path, exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)
    os.makedirs(so_data_folder_path, exist_ok=True)

    # Retrieve OmniPath GPs (source: ligand genes; target: receptor genes)
    omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
        species=species,
        load_from_disk=True,
        save_to_disk=False,
        lr_network_file_path=omnipath_lr_network_file_path,
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        plot_gp_gene_count_distributions=False)

    # Display example OmniPath GP
    omnipath_gp_names = list(omnipath_gp_dict.keys())
    random.shuffle(omnipath_gp_names)
    omnipath_gp_name = omnipath_gp_names[0]

    # Retrieve NicheNet GPs (source: ligand genes; target: receptor genes, target genes)
    nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
        species=species,
        version="v2",
        keep_target_genes_ratio=1.,
        max_n_target_genes_per_gp=250,
        load_from_disk=True,
        save_to_disk=False,
        lr_network_file_path=nichenet_lr_network_file_path,
        ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        plot_gp_gene_count_distributions=False)

    # Display example NicheNet GP
    nichenet_gp_names = list(nichenet_gp_dict.keys())
    random.shuffle(nichenet_gp_names)
    nichenet_gp_name = nichenet_gp_names[0]
    print(f"{nichenet_gp_name}: {nichenet_gp_dict[nichenet_gp_name]}")

    # Retrieve MEBOCOST GPs (source: enzyme genes; target: sensor genes)
    mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
        dir_path=mebocost_enzyme_sensor_interactions_folder_path,
        species=species,
        plot_gp_gene_count_distributions=False)

    # # Display example MEBOCOST GP
    # mebocost_gp_names = list(mebocost_gp_dict.keys())
    # random.shuffle(mebocost_gp_names)
    # mebocost_gp_name = mebocost_gp_names[0]
    # print(f"{mebocost_gp_name}: {mebocost_gp_dict[mebocost_gp_name]}")

    # Filter and combine GPs
    gp_dicts = [omnipath_gp_dict, nichenet_gp_dict, mebocost_gp_dict]
    combined_gp_dict = filter_and_combine_gp_dict_gps_v2(
        gp_dicts,
        verbose=True)

    print(f"Number of gene programs after filtering and combining: "
        f"{len(combined_gp_dict)}.")

    for adata_batch in ann_list:
        print("Computing spatial neighborhood graph...\n")
        sq.gr.spatial_neighbors(
            adata_batch,
            coord_type="generic",
            spatial_key=spatial_key,
            n_neighs=n_neighbors
        )
        # Make adjacency matrix symmetric
        adata_batch.obsp[adj_key] = adata_batch.obsp[adj_key].maximum(
            adata_batch.obsp[adj_key].T
        )
    adata = ad.concat(ann_list, join="inner")

    # Combine spatial neighborhood graphs as disconnected components
    batch_connectivities = []
    len_before_batch = 0
    for i in range(len(ann_list)):
        if i == 0: # first batch
            after_batch_connectivities_extension = sp.csr_matrix(
                (ann_list[0].shape[0],
                (adata.shape[0] -
                ann_list[0].shape[0])))
            batch_connectivities.append(sp.hstack(
                (ann_list[0].obsp[adj_key],
                after_batch_connectivities_extension)))
        elif i == (len(ann_list) - 1): # last batch
            before_batch_connectivities_extension = sp.csr_matrix(
                (ann_list[i].shape[0],
                (adata.shape[0] -
                ann_list[i].shape[0])))
            batch_connectivities.append(sp.hstack(
                (before_batch_connectivities_extension,
                ann_list[i].obsp[adj_key])))
        else: # middle batches
            before_batch_connectivities_extension = sp.csr_matrix(
                (ann_list[i].shape[0], len_before_batch))
            after_batch_connectivities_extension = sp.csr_matrix(
                (ann_list[i].shape[0],
                (adata.shape[0] -
                ann_list[i].shape[0] -
                len_before_batch)))
            batch_connectivities.append(sp.hstack(
                (before_batch_connectivities_extension,
                ann_list[i].obsp[adj_key],
                after_batch_connectivities_extension)))
        len_before_batch += ann_list[i].shape[0]
    adata.obsp[adj_key] = sp.vstack(batch_connectivities)

    add_gps_from_gp_dict_to_adata(
        gp_dict=combined_gp_dict,
        adata=adata,
        gp_targets_mask_key=gp_targets_mask_key,
        gp_targets_categories_mask_key=gp_targets_categories_mask_key,
        gp_sources_mask_key=gp_sources_mask_key,
        gp_sources_categories_mask_key=gp_sources_categories_mask_key,
        gp_names_key=gp_names_key,
        min_genes_per_gp=2,
        min_source_genes_per_gp=1,
        min_target_genes_per_gp=1,
        max_genes_per_gp=None,
        max_source_genes_per_gp=None,
        max_target_genes_per_gp=None)

    # Initialize model
    model = NicheCompass(adata,
                        counts_key=counts_key,
                        adj_key=adj_key,
                        cat_covariates_embeds_injection=cat_covariates_embeds_injection,
                        cat_covariates_keys=cat_covariates_keys,
                        cat_covariates_no_edges=cat_covariates_no_edges,
                        cat_covariates_embeds_nums=cat_covariates_embeds_nums,
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
                lambda_l1_masked=lambda_l1_masked,
                edge_batch_size=edge_batch_size,
                n_sampled_neighbors=n_sampled_neighbors,
                use_cuda_if_available=use_cuda_if_available,
                verbose=False)
    
    adata = model.adata
    adata.obsm['integrated'] = adata.obsm['nichecompass_latent']
    
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata