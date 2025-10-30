# src/gwinspect/convert.py
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Conversions between temperature T [GeV], energy scale E [GeV],
and present-day GW frequency f [Hz].

Relies on:
- bg_constant.py : T_0 (GeV), m_P, Omega_rad_0, BBN_constraint, A_S
- thermo.py      : g_star_k(T), g_s_k(T) (table ordered high→low T)

Public API:
    temp_to_energy(T)   -> (E, T, g_star(T))
    vec_temp_to_energy
    Temp(E)             -> T via prebuilt lookup table (T in [1e-3, 1e16] GeV)
    vec_Temp
    freq(T)             -> Hz
    energy_from_T(T)    -> GeV
    Num_of_e_folds(w,Ei,Ef)
"""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np

ArrayLike = Union[float, np.ndarray]

# ---- package imports ----
from .thermo import g_star_k, g_s_k
from .bg_constant import T_0  # GeV (CMB temperature today)

# --------------------------------------------------------------------------------------
# Core relations (kept as in your original formulas, vectorized)
# --------------------------------------------------------------------------------------

def energy_from_T(T: ArrayLike) -> np.ndarray:
    """T [GeV] → E [GeV].  E = T * (π² g*(T) / 30)^(1/4)   [Eq. (2.51)]"""
    T = np.asarray(T, dtype=float)
    g = g_star_k(T)
    return T * (np.pi**2 * g / 30.0) ** 0.25


def temp_to_energy(T: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (E[GeV], T[GeV], g_star(T)) for input temperature(s) T [GeV]."""
    T = np.asarray(T, dtype=float)
    g = g_star_k(T)
    E = (np.pi**2 / 30.0 * g) ** 0.25 * T
    return E, T, g

# Vectorized alias (API compatibility)
vec_temp_to_energy = np.vectorize(lambda t: temp_to_energy(t), otypes=[float, float, float])

# --------------------------------------------------------------------------------------
# Build lookup table for Temp(E) over 1e-3 .. 1e16 GeV (as in your original code)
# --------------------------------------------------------------------------------------

# Temperature grid (increasing)
_temp_grid = np.logspace(-3.0, 16.0, 100_000, dtype=float)  # GeV
_gstar_grid = g_star_k(_temp_grid)
_energy_grid = (_temp_grid * (np.pi**2 * _gstar_grid / 30.0) ** 0.25).astype(float)

# Columns: [E, T, g_star]
temperature_table = np.column_stack((_energy_grid, _temp_grid, _gstar_grid))

def _ceil_index_in_energy(E: np.ndarray) -> np.ndarray:
    """Index of the first table energy >= E on increasing grid."""
    idx = np.searchsorted(temperature_table[:, 0], E, side="left")
    return np.clip(idx, 0, temperature_table.shape[0] - 1)

def Temp(E: ArrayLike) -> np.ndarray:
    """E [GeV] → T [GeV] using prebuilt table (valid for T in [1e-3, 1e16] GeV)."""
    E = np.asarray(E, dtype=float)
    E_min = temperature_table[0, 0]
    E_max = temperature_table[-1, 0]
    if np.any(E < E_min) or np.any(E > E_max):
        raise ValueError(
            f"E is outside table range [{E_min:.3e}, {E_max:.3e}] GeV. "
            "Increase the T grid if needed."
        )
    idx = _ceil_index_in_energy(E)
    return temperature_table[idx, 1]

# Vectorized alias (API compatibility)
vec_Temp = np.vectorize(lambda e: Temp(e), otypes=[float])

# --------------------------------------------------------------------------------------
# Frequency and e-folds
# --------------------------------------------------------------------------------------

def freq(T: ArrayLike) -> np.ndarray:
    """T [GeV] → present GW frequency f [Hz].  [Eq. (2.50)]
       f = 7.43e-8 * (g_s(T0)/g_s(T))^(1/3) * (g*(T)/90)^(1/2) * T
       with T0 = T_0 (GeV) from bg_constant.py
    """
    T = np.asarray(T, dtype=float)
    gs_ratio = (g_s_k(T_0) / g_s_k(T)) ** (1.0 / 3.0)
    g_ratio = (g_star_k(T) / 90.0) ** 0.5
    return 7.43e-8 * gs_ratio * g_ratio * T


def Num_of_e_folds(w: float, Ei: float, Ef: float) -> float:
    """Number of e-folds for EoS w, from Ei → Ef (GeV)."""
    return 4.0 / (3.0 * (1.0 + w)) * float(np.log(Ei / Ef))

# --------------------------------------------------------------------------------------

__all__ = [
    "temp_to_energy",
    "vec_temp_to_energy",
    "temperature_table",
    "Temp",
    "vec_Temp",
    "freq",
    "energy_from_T",
    "Num_of_e_folds",
]
