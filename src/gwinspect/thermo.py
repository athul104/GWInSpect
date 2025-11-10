# src/gwinspect/thermo.py
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Thermodynamics utilities for effective relativistic degrees of freedom.

This module computes tabulated or fitted values of:
    - g_star(T): Effective degrees of freedom in energy density
    - g_s(T): Effective degrees of freedom in entropy density

It includes routines for loading data from `eff_rel_dof.txt`, computing
fitting functions, and generating updated tables.
"""

from __future__ import annotations
from typing import Union, Sequence

import numpy as np

# ----------------------------------------------------------------------------
# Data Loader
# ----------------------------------------------------------------------------

try:
    from importlib.resources import files as _files  # Python 3.9+
except Exception:
    _files = None

def load_eff_rel_dof() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the table of temperature, g_star, g_s, and energy values.

    Returns
    -------
    Temp_in_GeV : ndarray
        Temperature values [GeV].
    g_star_tab : ndarray
        Effective relativistic degrees of freedom in energy density.
    g_s_tab : ndarray
        Effective relativistic degrees of freedom in entropy density.
    Energy_in_GeV : ndarray
        Corresponding energy scales [GeV].
    """
    if _files is None:
        raise RuntimeError("importlib.resources is unavailable on this Python.")

    path = _files("gwinspect.data").joinpath("eff_rel_dof.txt")

    try:
        raw = np.loadtxt(str(path), dtype=float)
    except ValueError:
        raw = np.loadtxt(str(path), dtype=float, skiprows=1)

    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError("eff_rel_dof.txt must have 4 numeric columns.")

    Temp_in_GeV = raw[:, 0]
    g_star_tab = raw[:, 1]
    g_s_tab = raw[:, 2]
    energy_in_GeV = raw[:, 3]

    return Temp_in_GeV, g_star_tab, g_s_tab, energy_in_GeV

# ----------------------------------------------------------------------------
# Physical Mass Constants (GeV)
# ----------------------------------------------------------------------------
m_e = 511e-6
m_mu = 0.1056
m_pi0 = 0.135
m_pipm = 0.140
m_1, m_2, m_3, m_4 = 0.5, 0.77, 1.2, 2.0

ArrayLike = Union[float, Sequence[float], np.ndarray]

def _return_like_input(x, y):
    return y.item() if np.isscalar(x) else y

# ----------------------------------------------------------------------------
# Fitting functions
# ----------------------------------------------------------------------------
def f_rho(x):
    x = np.asarray(x)
    return _return_like_input(x, np.exp(-1.04855 * x) * (1 + 1.03757*x + 0.508630*x**2 + 0.0893988*x**3))

def b_rho(x):
    x = np.asarray(x)
    return _return_like_input(x, np.exp(-1.03149 * x) * (1 + 1.03317*x + 0.398264*x**2 + 0.0648056*x**3))

def f_s(x):
    x = np.asarray(x)
    return _return_like_input(x, np.exp(-1.04190 * x) * (1 + 1.03400*x + 0.456426*x**2 + 0.0595248*x**3))

def b_s(x):
    x = np.asarray(x)
    return _return_like_input(x, np.exp(-1.03365 * x) * (1 + 1.03397*x + 0.342548*x**2 + 0.0506182*x**3))

def S_fit(x):
    x = np.asarray(x)
    return _return_like_input(x, 1 + (7/4) * np.exp(-1.0419 * x) * (1 + 1.034*x + 0.456426*x**2 + 0.0595249*x**3))

# ----------------------------------------------------------------------------
# Fit Coefficients
# ----------------------------------------------------------------------------
a_arr = np.array([1.0, 1.11724, 0.312672, -0.0468049, -0.0265004, -0.0011976,
                  0.000182812, 0.000136436, 8.55051e-05, 1.2284e-05, 3.82259e-07, -6.87035e-09])

b_arr = np.array([0.0143382, 0.0137559, 0.00292108, -0.000538533, -0.000162496,
                  -2.87906e-05, -3.84278e-06, 2.78776e-06, 7.40342e-07, 1.1721e-07,
                  3.72499e-09, -6.74107e-11])

c_arr = np.array([1.0, 0.607869, -0.154485, -0.224034, -0.0282147, 0.029062,
                  0.00686778, -0.00100005, -0.000169104, 1.06301e-05,
                  1.69528e-06, -9.33311e-08])

d_arr = np.array([70.7388, 91.8011, 33.1892, -1.39779, -1.52558, -0.0197857,
                  -0.160146, 8.22615e-05, 0.0202651, -1.82134e-05,
                  7.83943e-05, 7.13518e-05])


def _poly_ratio(t, num, den):
    return np.polyval(num[::-1], t) / np.polyval(den[::-1], t)

# ----------------------------------------------------------------------------
# Main Functions
# ----------------------------------------------------------------------------
def g_star(T: ArrayLike) -> np.ndarray | float:
    """
    Compute the effective number of relativistic degrees of freedom in energy density, gₛₜₐᵣ(T),
    using an analytic fit consistent with lattice + perturbative results.

    Parameters
    ----------
    T : float or array-like
        Temperature(s) in GeV. Must be ≥ 0. For T > 1e16 GeV, the fit is not validated and returns NaN.

    Returns
    -------
    float or np.ndarray
        Corresponding gₛₜₐᵣ(T) values. Returns a float if the input is scalar; otherwise a NumPy array.


    Reference
    ---------
    K. Saikawa and S. Shirai,  
    "Primordial gravitational waves, precisely: The role of thermodynamics in the Standard Model",  
    JCAP 05 (2018) 035, [arXiv:1803.01038](https://arxiv.org/abs/1803.01038).
    """

    Tin = T
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T)

    mask_hi = (T >= 0.12) & (T <= 1e16)
    mask_lo = (T >= 0.0) & (T < 0.12)

    if mask_hi.any():
        t = np.log(T[mask_hi])
        out[mask_hi] = _poly_ratio(t, a_arr, b_arr)

    if mask_lo.any():
        Tl = T[mask_lo]
        out[mask_lo] = (
            2.030
            + 1.353 * (S_fit(m_e / Tl)) ** (4 / 3)
            + 3.495 * f_rho(m_e / Tl)
            + 3.446 * f_rho(m_mu / Tl)
            + 1.05  * b_rho(m_pi0 / Tl)
            + 2.08  * b_rho(m_pipm / Tl)
            + 4.165 * b_rho(m_1 / Tl)
            + 30.55 * b_rho(m_2 / Tl)
            + 89.4  * b_rho(m_3 / Tl)
            + 8209  * b_rho(m_4 / Tl)
        )

    out[T < 0] = np.nan
    out[T > 1e16] = np.nan
    return _return_like_input(Tin, out)

def g_s(T: ArrayLike) -> np.ndarray | float:
    """
    Compute the effective number of relativistic degrees of freedom in entropy density, gₛ(T),
    using an analytic fit consistent with lattice + perturbative results.

    Parameters
    ----------
    T : float or array-like
        Temperature(s) in GeV. Must be ≥ 0. For T > 1e16 GeV, the fit is not validated and returns NaN.

    Returns
    -------
    float or np.ndarray
        Corresponding gₛ(T) values. Returns a float if the input is scalar; otherwise a NumPy array.


    Reference
    ---------
    K. Saikawa and S. Shirai,  
    "Primordial gravitational waves, precisely: The role of thermodynamics in the Standard Model",  
    JCAP 05 (2018) 035, [arXiv:1803.01038](https://arxiv.org/abs/1803.01038).
    """

    Tin = T
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T)

    mask_hi = (T >= 0.12) & (T <= 1e16)
    mask_lo = (T >= 0.0) & (T < 0.12)

    if mask_hi.any():
        t = np.log(T[mask_hi])
        frac = _poly_ratio(t, c_arr, d_arr)
        out[mask_hi] = g_star(T[mask_hi]) / (1 + frac)

    if mask_lo.any():
        Tl = T[mask_lo]
        out[mask_lo] = (
            2.008
            + 1.923 * S_fit(m_e / Tl)
            + 3.442 * f_s(m_e / Tl)
            + 3.468 * f_s(m_mu / Tl)
            + 1.034 * b_s(m_pi0 / Tl)
            + 2.068 * b_s(m_pipm / Tl)
            + 4.16  * b_s(m_1 / Tl)
            + 30.55 * b_s(m_2 / Tl)
            + 90.0  * b_s(m_3 / Tl)
            + 6209  * b_s(m_4 / Tl)
        )

    out[T < 0] = np.nan
    out[T > 1e16] = np.nan
    return _return_like_input(Tin, out)




__all__ = [
    "g_star",
    "g_s",
    "load_eff_rel_dof",
]