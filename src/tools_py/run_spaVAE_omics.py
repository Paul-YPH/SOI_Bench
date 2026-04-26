import warnings
warnings.filterwarnings("ignore")

from time import time
import torch
from spaMultiVAE import SPAMULTIVAE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import scanpy as sc
from spaMultiVAE.preprocess import normalize, geneSelection
from sklearn.cluster import KMeans
import argparse
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering

def run_spaVAE_omics(adata, **args_dict):
    seed = args_dict['seed']
    clust = args_dict['clust']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    parser = argparse.ArgumentParser(description='Spatial dependency-aware variational autoencoder for spatial multi-omics data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--select_genes', default=0, type=int)
    parser.add_argument('--select_proteins', default=0, type=int)
    parser.add_argument('--batch_size', default="auto")
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--train_size', default=0.95, type=float)
    parser.add_argument('--patience', default=200, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--gene_noise', default=0, type=float)
    parser.add_argument('--protein_noise', default=0, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0, type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[128, 64], type=int)
    parser.add_argument('--GP_dim', default=2, type=int)
    parser.add_argument('--Normal_dim', default=18, type=int)
    parser.add_argument('--gene_decoder_layers', nargs="+", default=[128], type=int)
    parser.add_argument('--protein_decoder_layers', nargs="+", default=[128], type=int)
    parser.add_argument('--dynamicVAE', default=True, type=bool, 
                        help='whether to use dynamicVAE to tune the value of beta, if setting to false, then beta is fixed to initial value')
    parser.add_argument('--init_beta', default=10, type=float, help='initial coefficient of the KL loss')
    parser.add_argument('--min_beta', default=4, type=float, help='minimal coefficient of the KL loss')
    parser.add_argument('--max_beta', default=25, type=float, help='maximal coefficient of the KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--grid_inducing_points', default=True, type=bool, 
                        help='whether to generate grid inducing points or use k-means centroids on locations as inducing points')
    parser.add_argument('--inducing_point_steps', default=10, type=int)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=40, type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--final_latent_file', default='final_latent.txt')
    parser.add_argument('--gene_denoised_counts_file', default='gene_denoised_counts.txt')
    parser.add_argument('--protein_denoised_counts_file', default='protein_denoised_counts.txt')
    parser.add_argument('--protein_sigmoid_file', default='protein_sigmoid.txt')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    
    adata_omics1 = ann_list[config['sample_list'].tolist().index('rna')]
    adata_omics2 = ann_list[config['sample_list'].tolist().index('adt')]

    sc.pp.filter_genes(adata_omics2, min_counts=1) 
    sc.pp.filter_genes(adata_omics1, min_counts=1)

    sc.pp.filter_cells(adata_omics2, min_counts=1)
    sc.pp.filter_cells(adata_omics1, min_counts=1)

    adata_omics1.obs_names = [n.split('_', 1)[1] for n in adata_omics1.obs_names]
    adata_omics2.obs_names = [n.split('_', 1)[1] for n in adata_omics2.obs_names]
    
    common_index = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
    adata_omics1 = adata_omics1[common_index].copy()
    adata_omics2 = adata_omics2[common_index].copy()
    
    x1 = np.array(adata_omics1.X.toarray()).astype('float32')     # gene count matrix
    x2 = np.array(adata_omics2.X.toarray()).astype('float32')  # protein count matrix
    loc = np.array(adata_omics1.obsm['spatial']).astype('float32')       # location information
    loc_raw = loc.copy()    # location information
    
    del ann_list
    gc.collect()

    if args.batch_size == "auto":
        if x1.shape[0] <= 1024:
            args.batch_size = 128
        elif x1.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)
        
    if args.select_genes > 0:
        importantGenes = geneSelection(x1, n=args.select_genes, plot=False)
        x1 = x1[:, importantGenes]
        #np.savetxt("selected_genes.txt", importantGenes, delimiter=",", fmt="%i")

    if args.select_proteins > 0:
        importantProteins = geneSelection(x2, n=args.select_proteins, plot=False)
        x2 = x2[:, importantProteins]
        #np.savetxt("selected_proteins.txt", importantProteins, delimiter=",", fmt="%i")
        
    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * args.loc_range
    
    if args.grid_inducing_points:
        eps = 1e-5
        initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
        print(initial_inducing_points.shape)
    else:
        loc_kmeans = KMeans(n_clusters=args.inducing_point_nums, n_init=100).fit(loc)
        np.savetxt("location_centroids.txt", loc_kmeans.cluster_centers_, delimiter=",")
        np.savetxt("location_kmeans_labels.txt", loc_kmeans.labels_, delimiter=",", fmt="%i")
        initial_inducing_points = loc_kmeans.cluster_centers_

    adata1 = sc.AnnData(x1, dtype="float32")
    adata2 = sc.AnnData(x2, dtype="float32")
    adata1.obs = adata_omics1.obs
    adata2.obs = adata_omics2.obs
    del adata_omics1, adata_omics2
    
    adata1 = normalize(adata1,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    adata2 = normalize(adata2,
                      size_factors=False,
                      normalize_input=True,
                      logtrans_input=True)

    adata2_no_scale = sc.AnnData(x2, dtype="float32")
    adata2_no_scale = normalize(adata2_no_scale,
                      size_factors=False,
                      normalize_input=False,
                      logtrans_input=True)
    
    print(adata2_no_scale.X.shape)
    print("zero-variance proteins:", np.sum(np.var(adata2_no_scale.X, axis=0) < 1e-6))

    gm = GaussianMixture(n_components=2, covariance_type="diag", n_init=20, reg_covar=1e-4).fit(adata2_no_scale.X)
    back_idx = np.argmin(gm.means_, axis=0)
    protein_log_back_mean = np.log(np.expm1(gm.means_[back_idx, np.arange(adata2_no_scale.n_vars)]))
    protein_log_back_scale = np.sqrt(gm.covariances_[back_idx, np.arange(adata2_no_scale.n_vars)])
    print("protein_back_mean shape", protein_log_back_mean.shape)

    model = SPAMULTIVAE(gene_dim=adata1.n_vars, protein_dim=adata2.n_vars, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, 
        encoder_layers=args.encoder_layers, gene_decoder_layers=args.gene_decoder_layers, protein_decoder_layers=args.protein_decoder_layers,
        gene_noise=args.gene_noise, protein_noise=args.protein_noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata1.n_obs, KL_loss=args.KL_loss, dynamicVAE=args.dynamicVAE,
        init_beta=args.init_beta, min_beta=args.min_beta, max_beta=args.max_beta, protein_back_mean=protein_log_back_mean, protein_back_scale=protein_log_back_scale, 
        dtype=torch.float32, device=args.device)
    
    t0 = time()
    model.train_model(pos=loc, gene_ncounts=adata1.X, gene_raw_counts=adata1.raw.X, gene_size_factors=adata1.obs.size_factors, 
                protein_ncounts=adata2.X, protein_raw_counts=adata2.raw.X,
                lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file)
    print('Training time: %d seconds.' % int(time() - t0))
    
    final_latent = model.batching_latent_samples(X=loc, gene_Y=adata1.X, protein_Y=adata2.X, batch_size=args.batch_size)
    adata = adata1.copy()
    del adata1, adata2
    gc.collect()
    
    # Clustering
    print('### Clustering...')
    adata.obsm['integrated'] = final_latent
    adata.obsm['spatial'] = loc_raw
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust,mclust_version = "2")
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    adata.obsm['spatial'] = loc_raw

    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata