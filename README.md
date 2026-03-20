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
cd ~/Codes/mempost
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

#### Step 2: Run DPS posterior sampling

Config: `configs/helmholtz_dps_comparison_c64.json`.

```bash
python scripts/helmholtz_dps_comparison_c64.py --gpu_id 0 --seed 123
```

#### Step 3: Generate paper-quality panels

```bash
python scripts/generate_paper_panels.py
```

Output: `figs/helmholtz_panels/` with all panels for Figures 2--5 (true model, loss, mean, std, scatter, calibration, memorized pairs).

---

## Tests

```bash
pytest tests/ -v
```

## Author

Ali Siahkoohi (alisk@ucf.edu)
