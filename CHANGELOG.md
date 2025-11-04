# Changelog

## 1.0.0 — 2025-11-03  
**First stable release — verified modules, tested interfaces, PyPI-ready**

### Highlights:
- ✅ Verified all core modules with full end-to-end consistency.
- 📄 Updated `README.md` and `tutorial_gwinspect.ipynb` with usage and validation examples.
- 🧪 New `examples/` folder for demonstrating API behavior.

### Improvements:
- `constants.py`: Now user-overridable via `set_constants()`.
- `thermo.py`:
  - Supports user-defined relativistic degrees of freedom data (TXT, CSV, NumPy arrays).
  - Efficient floor-based interpolation.
- `cosmo_tools.py` (renamed from `convert.py`):
  - Renamed functions for clarity: `energy_of_T`, `temp_of_E`, `freq_of_T`, `get_efolds`.
  - Added docstrings, cleaner structure.
- `spectrum.py`:
  - Main function renamed to `compute_omega_gw` for naming clarity.
  - Added `f_custom` input to evaluate Ω_GW(f) at arbitrary frequencies.
  - Improved internal logic and numerical stability in coefficient propagation.
- `constraint.py`:
  - Improved BBN integral handling for special case `w = 1/3`.

### Packaging and Build:
- Version bumped to `1.0.0`.
- Cleaned and expanded `pyproject.toml`.
- Classification updated to `"Development Status :: 5 - Production/Stable"`.

---

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
