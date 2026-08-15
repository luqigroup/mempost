"""Fetch released inputs on first use.

Datasets are hosted publicly and downloaded automatically the first time a script
needs them; every fetch is a no-op once the file is on disk. Call ``ensure`` with a
repo-relative path and it returns the local path, downloading it if missing:

    from mempost.download import ensure
    path = ensure("data/seismic/dataset_train.h5")

The registry below maps each repo-relative path to a public direct-download URL. The
seismic archives are the Parihaka broadband-reflectivity dataset (key ``broadband_dm``)
the DDPM prior trains on; the raw Parihaka volume is not redistributed.
"""
from __future__ import annotations

import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# repo-relative path -> (tier, direct-download URL)
REGISTRY: dict[str, tuple[str, str]] = {
    "data/seismic/dataset_train.h5": (
        "data",
        "https://www.dropbox.com/scl/fi/j2rkgfl9wa9f53hmo4cts/"
        "seismic_train_lam3.5_v2.h5?rlkey=cntcpqsy1cjfdebhcrkw3mm83&dl=1",
    ),
    "data/seismic/dataset_eval.h5": (
        "data",
        "https://www.dropbox.com/scl/fi/rlt4791uwwref6wybxxal/"
        "seismic_eval_lam3.5_v2.h5?rlkey=opo1f8unebxgeqtwkflr2mmvh&dl=1",
    ),
}


def _hook(block: int, size: int, total: int) -> None:
    if total > 0:
        pct = min(100, 100 * block * size // total)
        sys.stdout.write(f"\r  {pct:3d}%  ({block * size / 1e6:.0f} MB)")
        sys.stdout.flush()


def _ensure_one(rel: str) -> str:
    path = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
    if os.path.exists(path):
        return path

    entry = REGISTRY.get(rel)
    if entry is None:
        raise RuntimeError(
            f"{rel} is missing and is not a released artifact; regenerate it with the "
            f"producer script named for it in the README."
        )
    _, url = entry
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[mempost] {rel} not found; downloading once from the public mirror.")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, _hook)
        sys.stdout.write("\n")
        with open(tmp, "rb") as f:  # an expired link returns an HTML page, not the file
            if f.read(1) == b"<":
                raise RuntimeError(
                    f"the link for {rel} returned a web page, not the file; "
                    f"see the Data section of the README."
                )
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)
    print(f"[mempost] saved {rel} ({os.path.getsize(path) / 1e6:.0f} MB)")
    return path


def ensure(*rel_paths: str) -> str | tuple[str, ...]:
    """Return local paths for ``rel_paths``, downloading any that are missing."""
    out = tuple(_ensure_one(p) for p in rel_paths)
    return out[0] if len(out) == 1 else out


def ensure_tier(tier: str) -> None:
    """Download every registered artifact in one tier ahead of time."""
    names = [k for k, (t, _) in REGISTRY.items() if t == tier]
    if not names:
        tiers = sorted({t for t, _ in REGISTRY.values()})
        raise RuntimeError(f"unknown tier {tier!r}; known tiers: {tiers}")
    for n in sorted(names):
        _ensure_one(n)
