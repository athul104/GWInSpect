# src/gwinspect/cosmo_tools.py
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Conversions between temperature T [GeV], energy scale E [GeV],
and present-day GW frequency f [Hz].

Relies on:
- constants.py : T0 (GeV), m_P, omega_rad0, A_S
- thermo.py    : g_star(T), g_s(T)

Public API:
    temp_of_E(E)         -> T from tabulated inverse energy relation
    freq_of_T(T)         -> Hz
    energy_of_T(T)       -> GeV
    compute_efolds(w, Ei, Ef) -> e-fold count
"""

from __future__ import annotations

import numpy as np
from typing import Union, Sequence

from .thermo import g_star, g_s, load_eff_rel_dof
from .constants import T0  # CMB temperature today (GeV)

ArrayLike = Union[float, Sequence[float], np.ndarray]

def energy_of_T(T: ArrayLike) -> np.ndarray | float:
    """
    Convert temperature [GeV] to energy scale [GeV].

    Parameters
    ----------
    T : float or array-like
        Temperature(s) in GeV.

    Returns
    -------
    float or np.ndarray
        Corresponding energy scale(s).
    """
    is_scalar = np.isscalar(T)
    T = np.asarray(T, dtype=float)
    g = g_star(T)
    E = T * (np.pi**2 * g / 30.0) ** 0.25
    return float(E) if is_scalar else E

def _get_energy_grid():
    """
    Load (T, E) from a stored eff_rel_dof.txt in data.
    temp_grid = np.logspace(-13, 16, 1000, base = 10)
    energy_grid = energy_of_T(temp_grid)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Temperature and corresponding energy arrays.
    """

    temp_grid, _, _, energy_grid = load_eff_rel_dof()

    return temp_grid, energy_grid

def _floor_index_in_energy(E: np.ndarray) -> np.ndarray:
    """
    Return the index of the largest tabulated E <= query E.

    Parameters
    ----------
    E : array-like
        Energy values [GeV].

    Returns
    -------
    np.ndarray
        Floor indices.
    """
    temp_grid, energy_grid = _get_energy_grid()
    E_asc = np.atleast_1d(np.asarray(E, dtype=float))
    n = E_asc.size
    idx_asc = np.searchsorted(E_asc, E, side="right") - 1
    idx_asc = np.clip(idx_asc, 0, n - 1)
    return idx_asc

def temp_of_E(E: ArrayLike) -> np.ndarray | float:
    """
    Convert energy scale [GeV] to temperature [GeV].
    Uses inverse of energy_of_T(T) with tabulated floor matching from data/eff_rel_dof.txt.

    Parameters
    ----------
    E : float or array-like
        Energy scale(s) in GeV.

    Returns
    -------
    float or np.ndarray
        Corresponding temperature(s) in GeV.
    """
    is_scalar = np.isscalar(E)
    E = np.asarray(E, dtype=float)
    temp_grid, _ = _get_energy_grid()
    idx = _floor_index_in_energy(E)
    floor_temp = temp_grid[idx]
    g_star_val = g_star(floor_temp)
    T_out = E / ((np.pi**2 * g_star_val / 30.0) ** 0.25)
    return float(T_out) if is_scalar else T_out

def freq_of_T(T: ArrayLike) -> np.ndarray | float:
    """
    Convert temperature [GeV] to present-day GW frequency [Hz].

    Parameters
    ----------
    T : float or array-like
        Temperature(s) in GeV.

    Returns
    -------
    float or np.ndarray
        Present-day GW frequency in Hz.
    """
    is_scalar = np.isscalar(T)
    T = np.asarray(T, dtype=float)
    gs_ratio = (g_s(T0) / g_s(T)) ** (1.0 / 3.0)
    g_ratio = (g_star(T) / 90.0) ** 0.5
    freq = 7.43e-8 * gs_ratio * g_ratio * T
    return float(freq) if is_scalar else freq

def compute_efolds(w: float, Ei: float, Ef: float) -> float:
    """
    Compute number of e-folds for a constant EoS w from energy scale Ei → Ef.

    Parameters
    ----------
    w : float
        Equation of state parameter.
    Ei : float
        Initial energy scale [GeV].
    Ef : float
        Final energy scale [GeV].

    Returns
    -------
    float
        Number of e-folds.
    """
    return 4.0 / (3.0 * (1.0 + w)) * float(np.log(Ei / Ef))

__all__ = [
    "temp_of_E",
    "freq_of_T",
    "energy_of_T",
    "compute_efolds",
]
