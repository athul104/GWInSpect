# Changelog

## 0.1.0 — 2025-10-30
**Initial public WIP**
- Packaging: `pyproject.toml` (hatchling), GPL-3.0-or-later.
- Data: bundled `src/gwinspect/data/eff_rel_dof.txt`.
- Modules:
  - `thermo.py`: `load_eff_rel_dof`, `g_star_k`, `g_s_k`.
  - `convert.py`: `freq`, `energy_from_T`, `Temp`, e-folds helpers.
  - `spectrum.py`: `get_omega_gw` (piecewise reheating; mpmath BesselJ; SciPy gamma).
  - `constraint.py`: `bbn_bound` (piecewise BBN integral; proper `w=1/3` log-limit).
- Public API re-exports in `gwinspect.__init__`.
- Docs: initial `README.md` (WIP), added `CITATION.cff`.
