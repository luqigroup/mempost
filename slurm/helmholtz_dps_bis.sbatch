#!/bin/bash
#SBATCH --job-name=helm-dps-sips
#SBATCH --output=logs/helmholtz_dps_%j.out
#SBATCH --error=logs/helmholtz_dps_%j.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=60:00:00

module load gcc/gcc-12.2.0

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate mempost
cd /lustre/fs1/home/da389032/mempost
export PYTHONPATH=/lustre/fs1/home/da389032/mempost:$PYTHONPATH  


OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python -u scripts/helmholtz_dps_comparison_c64_bis.py \
    --num_samples 256 \
    --noise_rel 0.055 \
    --step_size 0.07 \
    --no_grad_norm 0 \
    --n_workers 16 \
    --true_model file \
    --true_model_path /lustre/fs1/home/da389032/mempost/data/avg3_true_model.npz \
    --z_lim_min -18 \
    --z_lim_max 20 \
    --upload 1