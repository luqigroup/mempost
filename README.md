# `mempost` - Posterior under memorized diffusion priors

Analytical and numerical experiments demonstrating how memorized score-based diffusion priors affect posterior inference in Bayesian inverse problems.

**Paper**: *On the role of memorization in learned priors for geophysical inverse problems* (IMAGE 2026)

## Overview

When a diffusion model memorizes its finite training set, the learned prior collapses to a Gaussian mixture. We derive the resulting posterior in closed form (a linearized Gaussian mixture) and show that its support shrinks to the training examples as the diffusion bandwidth vanishes. The Helmholtz full waveform inversion experiments confirm these predictions numerically via diffusion posterior sampling.

## Installation

```bash
conda create --name mempost "python<=3.12"
conda activate mempost

git clone https://github.com/alisiahkoohi/mempost
cd mempost
pip install -e .
```

**External dependencies:**

```bash
pip install -e "git+https://github.com/luqigroup/grf#egg=grf"
pip install -e "git+https://github.com/luqigroup/projorg#egg=projorg"
```

## Project structure

```
mempost/
├── mempost/
│   ├── models/                  # UNet1d score model, noise scheduler
│   ├── utils/
│   │   ├── gmm.py               # GMM prior/posterior math (Eqs. 3--7)
│   │   ├── helmholtz_c64.py     # 2D Helmholtz PDE solver (complex64, PML)
│   │   ├── helmholtz.py         # 2D Helmholtz PDE solver (complex128)
│   │   ├── kl_prior.py          # Karhunen--Loève velocity parameterization
│   │   ├── memorization_metrics.py  # Nearest-neighbor memorization ratio
│   │   └── normalizer.py        # Z-score normalization
│   └── plotting.py              # Visualization utilities
├── scripts/                     # Training and evaluation
├── configs/                     # JSON experiment configurations
└── tests/                       # Unit tests (pytest)
```

## Paper figures

| Figure | Description | Script |
|---|---|---|
| Figure 1 (a--c) | 1D posterior collapse at sigma = {0.5, 0.3, 0.05} | `stylized_gmm.py` |
| Figure 1 (d--f) | 2D posterior with linearized Gaussian mixture components | `stylized_gmm.py` |
| Figure 2 | Most memorized N=50 prior/posterior samples and nearest training neighbors | `helmholtz_dps_comparison_c64.py` |
| Figure 3 | DPS posterior analysis: true model, loss, calibration, mean, std, scatter (N=50, 200, 1000) | `helmholtz_dps_comparison_c64.py` |
| Table 1 | Memorization rates across N | `helmholtz_dps_comparison_c64.py` |

## Reproducing paper figures

All commands assume:
```bash
conda activate mempost
cd mempost
```

### Figure 1: Stylized GMM posterior collapse (no GPU)

Config: `configs/stylized_gmm.json`.

```bash
python scripts/stylized_gmm.py
```

Output: `figs/stylized_panels/1d_sigma*.png` and `figs/stylized_panels/2d_sigma*.png`.

---

### Figures 2--3 and Table 1: Helmholtz DPS comparison

This experiment requires trained score models (one per N in {50, 200, 1000}) and a GPU.

#### Step 1: Train score models (one per N)

Configs: `configs/helmholtz_fwi_N{50,200,1000}_overfit.json`.

```bash
python scripts/helmholtz_fwi.py --config configs/helmholtz_fwi_N50_overfit.json --gpu_id 0
python scripts/helmholtz_fwi.py --config configs/helmholtz_fwi_N200_overfit.json --gpu_id 0
python scripts/helmholtz_fwi.py --config configs/helmholtz_fwi_N1000_overfit.json --gpu_id 0
```

##### Pre-trained checkpoints (skip Step 1)

If you do not want to retrain, download the final-epoch checkpoints from Dropbox into the exact paths the DPS script reads:

```bash
D50="helmholtz_fwi_kl_K-10_grid_size-64_v_background-2.0_sigma_m-0.02_npml-10_pml_max-100.0_n_src-5_n_rec-40_frequency-4.0_hidden_dim-2048_nlayers-12_emb_size-256_nt-200_batchsize-50_max_epochs-50000_save_freq-5000_lr-0.0005_lr_final-5e-05_num_train-50_seed-42"
D200="helmholtz_fwi_kl_K-10_grid_size-64_v_background-2.0_sigma_m-0.02_npml-10_pml_max-100.0_n_src-5_n_rec-40_frequency-4.0_hidden_dim-2048_nlayers-12_emb_size-256_nt-200_batchsize-50_max_epochs-12500_save_freq-1250_lr-0.59b245396ff9fa352b4d3c414aeabb6b2840626c"
D1000="helmholtz_fwi_kl_K-10_grid_size-64_v_background-2.0_sigma_m-0.02_npml-10_pml_max-100.0_n_src-5_n_rec-40_frequency-4.0_hidden_dim-2048_nlayers-12_emb_size-256_nt-200_batchsize-50_max_epochs-2500_save_freq-250_lr-0.0005_lr_final-5e-05_num_train-1000_seed-42"

mkdir -p "data/checkpoints/$D50" "data/checkpoints/$D200" "data/checkpoints/$D1000"

wget 'https://www.dropbox.com/scl/fi/eed8ne65vz750iubm6g2d/checkpoint_49999.pth?rlkey=l6qbir3k1nkshdy6jljfjfsak&dl=1' \
    -O "data/checkpoints/$D50/checkpoint_49999.pth"
wget 'https://www.dropbox.com/scl/fi/kgkzbw7hb84j9rxoevull/checkpoint_12499.pth?rlkey=9zaax2pth9k17hicwem88y8d0&dl=1' \
    -O "data/checkpoints/$D200/checkpoint_12499.pth"
wget 'https://www.dropbox.com/scl/fi/6vivb70490zvclxtyp1qd/checkpoint_2499.pth?rlkey=6t1r8rbf9kon1q4xvxn9j5qsn&dl=1' \
    -O "data/checkpoints/$D1000/checkpoint_2499.pth"
```

Each file is ~784 MB. Run from the repo root.

#### Step 2: Run DPS posterior sampling

Config: `configs/helmholtz_dps_comparison_c64.json`.

```bash
python scripts/helmholtz_dps_comparison_c64.py --gpu_id 0 --seed 123
```

#### Step 3: Generate paper-quality panels

```bash
python scripts/generate_paper_panels.py
```

Output: `figs/helmholtz_panels/` with all panels for Figures 2--3, Table 1 (true model, loss, mean, std, scatter, calibration, memorized pairs).

---

## Tests

```bash
pytest tests/ -v
```

## Acknowledgments

Parts of this codebase were developed with the assistance of [Claude](https://claude.ai/) (Anthropic).

## Author

Ali Siahkoohi (alisk@ucf.edu)
