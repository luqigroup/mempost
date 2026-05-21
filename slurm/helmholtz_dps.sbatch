#!/bin/bash
#SBATCH --job-name=helm-dps
#SBATCH --output=logs/helmholtz_dps_%j.out
#SBATCH --error=logs/helmholtz_dps_%j.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=60:00:00

module load gcc/gcc-12.2.0

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate sips
cd $HOME/Codes/mempost

# Checkpoints must be pre-placed at $HOME/Codes/mempost/data/checkpoints/<dir>/checkpoint_<last>.pth
# See README "Pre-trained checkpoints" section for the wget commands.

# --- Avg-of-3 memorized true model, paper run ---
# True model = average of 3 most memorized N=50 training samples (indices 9, 8, 24).
# Gradient normalization ENABLED. step_size=0.07 is the published Figure 4 setting.

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python -u scripts/helmholtz_dps_comparison_c64.py \
    --num_samples 256 \
    --noise_rel 0.055 \
    --step_size 0.07 \
    --no_grad_norm 0 \
    --n_workers 16 \
    --true_model file \
    --true_model_path $HOME/Codes/mempost/data/avg3_true_model.npz \
    --z_lim_min -18 \
    --z_lim_max 20 \
    --upload 1
