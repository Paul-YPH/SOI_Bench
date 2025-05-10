#!/bin/bash
#SBATCH --job-name=D1_slat
#SBATCH --partition=mulan-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_scripts/D1_slat_%j.out
export CUDA_VISIBLE_DEVICES=$(echo $SLURM_JOB_GPUS | tr ',' '
' | head -n 1)
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
/net/mulan/home/penghuy/anaconda3/envs/scslat/bin/python /net/mulan/home/penghuy/benchmark/methods_v1/benchmarking.py \
  --input_path /net/mulan/home/penghuy/benchmark/data/D1 \
  --output_path /net/mulan/home/penghuy/benchmark/output/D1/SLAT \
  --metrics pattern mapping matching r_transform \
  --rigid True \
  --rep 1 \
  --tool_name SLAT  --sample 151673 151674 --multi_slice False --cluster_option mclust     --subsample True