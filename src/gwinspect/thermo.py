# src/gwinspect/thermo.py
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Load g*(T) and g*_s(T) tables and provide floor-based lookups.

Data file (packaged with the wheel):
    gwinspect/data/eff_rel_dof.txt

Format:
    Three numeric columns: T_GeV, g_star(T), g_s(T)
Order:
    Temperatures listed from HIGH to LOW (descending). We KEEP this order.

API:
    load_eff_rel_dof() -> (Temp_in_GeV, g_star_tab, g_s_tab)
    g_star_k(T)        -> value at the largest tabulated T <= query T
    g_s_k(T)           -> value at the largest tabulated T <= query T
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np

try:
    from importlib.resources import files as _files  # Python 3.9+
except Exception:  # pragma: no cover
    _files = None


@lru_cache(maxsize=None)
def load_eff_rel_dof() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the raw arrays in the file's ORIGINAL order (high→low T).

    Returns
    -------
    Temp_in_GeV : (N,) float ndarray
    g_star_tab  : (N,) float ndarray
    g_s_tab     : (N,) float ndarray
    """
    if _files is None:
        raise RuntimeError("importlib.resources is unavailable on this Python.")

    path = _files("gwinspect.data").joinpath("eff_rel_dof.txt")

    # Try with header; fall back to plain numeric text.
    try:
        arr = np.genfromtxt(str(path), dtype=float, names=True)
        cols = arr.dtype.names
        Temp_in_GeV = np.asarray(arr[cols[0]], dtype=float)
        g_star_tab  = np.asarray(arr[cols[1]], dtype=float)
        g_s_tab     = np.asarray(arr[cols[2]], dtype=float)
    except Exception:
        raw = np.loadtxt(str(path), dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 3:
            raise ValueError("eff_rel_dof.txt must have at least 3 numeric columns.")
        Temp_in_GeV = raw[:, 0].astype(float)
        g_star_tab  = raw[:, 1].astype(float)
        g_s_tab     = raw[:, 2].astype(float)

    # Basic sanity checks only (no sorting, no reordering)
    if np.any(~np.isfinite(Temp_in_GeV)) or np.any(Temp_in_GeV <= 0):
        raise ValueError("Temperature column must be positive, finite (GeV).")
    if np.any(~np.isfinite(g_star_tab)) or np.any(~np.isfinite(g_s_tab)):
        raise ValueError("g* columns contain non-finite values.")

    return Temp_in_GeV, g_star_tab, g_s_tab


def _floor_indices_desc(T_desc: np.ndarray, Tq_array: np.ndarray) -> np.ndarray:
    """
    For a DESCENDING temperature grid T_desc, return indices of the
    largest tabulated T <= each query Tq (floor in physical T).

    If Tq > max(T_desc): index 0 (first row).
    If Tq < min(T_desc): index N-1 (last row).
    """
    # Build an ascending view (no copies of the original arrays)
    T_asc = T_desc[::-1]
    n = T_asc.size

    # For ascending arrays, floor index = searchsorted(..., 'right') - 1
    idx_asc = np.searchsorted(T_asc, Tq_array, side="right") - 1
    # Clamp to [0, n-1]
    idx_asc = np.clip(idx_asc, 0, n - 1)
    # Map back to the original descending indices
    idx_desc = (n - 1) - idx_asc
    return idx_desc


def g_star_k(T: np.ndarray | float) -> np.ndarray | float:
    """
    Stepwise value of g_star(T) at the largest tabulated T <= query T.

    Parameters
    ----------
    T : float or array-like (GeV)

    Returns
    -------
    float or np.ndarray (matching T's shape)
    """
    Temp_in_GeV, g_star_tab, _ = load_eff_rel_dof()

    Tq = np.atleast_1d(np.asarray(T, dtype=float))
    idx = _floor_indices_desc(Temp_in_GeV, Tq)
    out = g_star_tab[idx]
    return float(out[0]) if np.isscalar(T) else out


def g_s_k(T: np.ndarray | float) -> np.ndarray | float:
    """
    Stepwise value of g_s(T) at the largest tabulated T <= query T.

    Parameters
    ----------
    T : float or array-like (GeV)

    Returns
    -------
    float or np.ndarray (matching T's shape)
    """
    Temp_in_GeV, _, g_s_tab = load_eff_rel_dof()

    Tq = np.atleast_1d(np.asarray(T, dtype=float))
    idx = _floor_indices_desc(Temp_in_GeV, Tq)
    out = g_s_tab[idx]
    return float(out[0]) if np.isscalar(T) else out


__all__ = ["load_eff_rel_dof", "g_star_k", "g_s_k"]
