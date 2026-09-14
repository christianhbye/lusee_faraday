# Archived notebooks

`faraday_sims.ipynb` (and `faraday_sims.py`, its jupytext export) drove
the paper's original two-port figures through `lusee_faraday.sim`. That
module — along with `beam.py`, `fast_sim.py`, `healpix.py`,
`rotations.py`, `utils.py` and `plot.py` — was retired when the analysis
moved onto luseepy + croissant, so **this notebook no longer runs**. It
is kept as the provenance record for the figures it produced, not as
working code.

It had in fact already stopped running earlier in that refactor, when
`spectrometer.py` was replaced by `lusee_faraday.channelization`; the
retirement of the two-port stack only made the breakage total.

Where the lineage survives:

- `scripts/compare_main_vs_asbuilt.py` — the Fig. 4 comparison, with its
  own self-contained `MainBeam`.
- `lusee_faraday.response.two_port_pair_alms` — the symmetric
  pseudo-dipole (two-port) arm, now through croissant.
- `scripts/step1_*.py`, `scripts/step_ionly.py` — the four-port analyses
  on the new stack.

## Notebooks deliberately left in `notebooks/`

Five other notebooks also reference the retired API:
`point_source-LN.ipynb`, `paper_plots.ipynb`, `wmap-time.ipynb`,
`faraday_analysis.ipynb`, `wmap_one.ipynb`. They are the author's own
working record rather than pipeline code, so moving or deleting them is
the author's call and not the refactor's. They stay where they are, and
they do not run either.
