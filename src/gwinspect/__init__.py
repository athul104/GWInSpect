# SPDX-License-Identifier: GPL-3.0-or-later
"""
gwinspect – reheating-era GW spectra & BBN constraint (WIP).

Public API (stable-ish while WIP):
- thermo:   load_eff_rel_dof, g_star_k, g_s_k
- convert:  energy_from_T, temp_to_energy, vec_temp_to_energy,
            Temp, vec_Temp, freq, Num_of_e_folds
- spectrum: get_omega_gw
- constraint: bbn_bound
"""

try:
    from .__about__ import __version__
except Exception:  # pragma: no cover
    __version__ = "0.1.0"

from .thermo import load_eff_rel_dof, g_star_k, g_s_k
from .convert import (
    energy_from_T,
    temp_to_energy, vec_temp_to_energy,
    Temp, vec_Temp,
    freq,
    Num_of_e_folds,
)
from .spectrum import get_omega_gw
from .constraint import bbn_bound

__all__ = [
    "__version__",
    # thermo
    "load_eff_rel_dof", "g_star_k", "g_s_k",
    # convert
    "energy_from_T", "temp_to_energy", "vec_temp_to_energy",
    "Temp", "vec_Temp", "freq", "Num_of_e_folds",
    # spectrum
    "get_omega_gw",
    # constraint
    "bbn_bound",
]
