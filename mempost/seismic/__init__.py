"""Seismic reflectivity prior: sampling, memorization scoring, data loading, and (later) the
linearized-Born DPS operator.

Leaf package for the realistic-seismic memorization track.  Modules are imported by full path
(``from mempost.seismic.sampling import sample_prior``) so ``diffusers`` / ``devito`` stay optional
and the KL/Helmholtz pipeline never imports them.
"""
