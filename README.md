# mempost

Code and data for

> **On the role of memorization in learned priors for geophysical inverse problems.**
> Ali Siahkoohi and Davide Sabeddu. IMAGE 2026.

## Overview

When a score-based diffusion model memorizes its finite training set, the learned prior
collapses to a Gaussian mixture centered on the training examples. We derive the resulting
posterior in closed form — a linearized Gaussian mixture — and show that its support shrinks
onto the training set as the diffusion bandwidth vanishes. Two inverse-problem experiments
confirm the prediction via diffusion posterior sampling (DPS).

| track | what it shows | needs |
|---|---|---|
| stylized GMM | the closed-form memorized prior, posterior, and its collapse as the bandwidth vanishes | CPU |
| Helmholtz FWI | posterior collapse onto training velocity models as the training set shrinks | CPU/GPU |
| seismic Born (Parihaka) | the same collapse on broadband reflectivity, with a realistic linearized-Born operator | GPU + [Devito] |

## Installation

```bash
git clone https://github.com/alisiahkoohi/mempost
cd mempost
pip install -e ".[seismic]"          # drop [seismic] to skip the Devito Born operator
```

Two path/experiment helpers are installed from source:

```bash
pip install -e "git+https://github.com/luqigroup/projorg#egg=projorg"
pip install -e "git+https://github.com/luqigroup/grf#egg=grf"
```

Python 3.9+ with PyTorch. A GPU speeds up prior training and DPS but is not needed for the
stylized figures; the Devito Born solves in the seismic track are CPU-only.

## Data

The seismic track uses the Parihaka broadband-reflectivity archive (HDF5, key `broadband_dm`).
It is hosted publicly and **downloaded automatically the first time a script needs it** — every
fetch is a no-op once the file is on disk. The registry lives in `mempost/download.py`; the two
archives are:

- `data/seismic/dataset_train.h5` — https://www.dropbox.com/scl/fi/j2rkgfl9wa9f53hmo4cts/seismic_train_lam3.5_v2.h5?rlkey=cntcpqsy1cjfdebhcrkw3mm83&dl=1
- `data/seismic/dataset_eval.h5` — https://www.dropbox.com/scl/fi/rlt4791uwwref6wybxxal/seismic_eval_lam3.5_v2.h5?rlkey=opo1f8unebxgeqtwkflr2mmvh&dl=1

To pre-fetch instead of letting them stream in on first use:

```python
from mempost.download import ensure_tier
ensure_tier("data")
```

The raw Parihaka volume is not redistributed; only the derived reflectivity archives are hosted.

## Reproducing the figures

**Stylized GMM** (CPU, seconds):

```bash
python scripts/stylized_gmm.py        # closed-form prior/posterior panels + the sigma-collapse sweep
python scripts/memorization_kde.py    # the KDE and delta-function memorization illustrations
```

**Helmholtz FWI** — train one overfit prior per training-set size, run DPS, then render the panels:

```bash
python scripts/helmholtz_fwi.py --config configs/helmholtz_fwi_N50_overfit.json   --gpu_id 0
python scripts/helmholtz_fwi.py --config configs/helmholtz_fwi_N200_overfit.json  --gpu_id 0
python scripts/helmholtz_fwi.py --config configs/helmholtz_fwi_N1000_overfit.json --gpu_id 0
python scripts/helmholtz_dps_comparison_c64.py --gpu_id 0
python scripts/generate_paper_panels.py
```

**Seismic Born (Parihaka)** — the dataset downloads on first use; train the prior, run DPS,
then render the talk figures:

```bash
python scripts/seismic_prior_ddpm.py --gpu_id 0                       # train the DDPM reflectivity prior
python scripts/seismic_dps_posterior.py --phase run --N 16   --truth-mode train-img --truth-idx 0
python scripts/seismic_dps_posterior.py --phase run --N 2048 --truth-mode train-img --truth-idx 0
python scripts/plot_talk_setup.py        # acquisition + two-prior setup
python scripts/plot_talk_singles.py      # reflectivity and posterior-std singles
python scripts/plot_talk_corr_distance.py
python scripts/plot_talk_loss_decomp.py
```

Figures are written under `figs/`; DPS caches under `plots/`. Both are regenerated, not tracked.

## Tests

```bash
pytest
```

The Helmholtz solver tests check grid convergence of the complex64 / complex128 solvers against
each other and the adjoint gradient against finite differences.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Ali Siahkoohi — alisk@ucf.edu
