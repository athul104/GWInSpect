# src/gwinspect/thermo.py
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Load g*(T) and g_s(T) tables and provide floor-based lookups.

Data file (packaged with the wheel):
    gwinspect/data/eff_rel_dof.txt

Format:
    Three numeric columns: T [GeV], g_star(T), g_s(T)
Order:
    Temperatures listed from HIGH to LOW (descending). We KEEP this order.

API:
    load_eff_rel_dof() -> (Temp_in_GeV, g_star_tab, g_s_tab)
    g_star(T)           -> value at the largest tabulated T <= query T
    g_s(T)              -> value at the largest tabulated T <= query T
    set_custom_eff_rel_dof(...)  -> provide user-defined data
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np
import os

try:
    from importlib.resources import files as _files  # Python 3.9+
except Exception:
    _files = None

# Global variable to hold custom data, if set
_custom_eff_rel_dof: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def set_custom_eff_rel_dof(data: str | np.ndarray | list | tuple) -> None:
    """
    Register a user-supplied table of [T, g_star, g_s].

    Parameters
    ----------
    data : str or array-like
        Either a file path (to .txt or .csv) or a NumPy array/nested list.
        Must be of shape (N, 3), where columns are T [GeV], g_star(T), g_s(T).
        Temperatures must be strictly positive, descending order preferred.
    """
    global _custom_eff_rel_dof

    if isinstance(data, str):
        if not os.path.isfile(data):
            raise FileNotFoundError(f"No such file: {data}")
        try:
            raw = np.loadtxt(data, dtype=float)
        except ValueError:
            try:
                raw = np.loadtxt(data, dtype=float, skiprows=1)
            except Exception as e:
                raise ValueError(f"Could not parse file with or without header: {e}")
    else:
        raw = np.asarray(data, dtype=float)

    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("Input must be a 2D array with exactly 3 columns: T, g_star, g_s")

    T, g1, g2 = raw[:, 0], raw[:, 1], raw[:, 2]
    if np.any(T <= 0) or np.any(~np.isfinite(T)) or np.any(~np.isfinite(g1)) or np.any(~np.isfinite(g2)):
        raise ValueError("All values must be finite; temperatures must be > 0.")

    _custom_eff_rel_dof = (T, g1, g2)
    load_eff_rel_dof.cache_clear()


@lru_cache(maxsize=None)
def load_eff_rel_dof() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the [T, g_star, g_s] table, either from built-in data or user-supplied values.

    Returns
    -------
    Temp_in_GeV : (N,) float ndarray
    g_star_tab  : (N,) float ndarray
    g_s_tab     : (N,) float ndarray
    """
    global _custom_eff_rel_dof

    if _custom_eff_rel_dof is not None:
        return _custom_eff_rel_dof

    if _files is None:
        raise RuntimeError("importlib.resources is unavailable on this Python.")

    path = _files("gwinspect.data").joinpath("eff_rel_dof.txt")

    try:
        raw = np.loadtxt(str(path), dtype=float)
    except ValueError:
        raw = np.loadtxt(str(path), dtype=float, skiprows=1)

    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError("eff_rel_dof.txt must have at least 3 numeric columns.")

    Temp_in_GeV = raw[:, 0].astype(float)
    g_star_tab  = raw[:, 1].astype(float)
    g_s_tab     = raw[:, 2].astype(float)

    if np.any(~np.isfinite(Temp_in_GeV)) or np.any(Temp_in_GeV <= 0):
        raise ValueError("Temperature column must be positive, finite (GeV).")
    if np.any(~np.isfinite(g_star_tab)) or np.any(~np.isfinite(g_s_tab)):
        raise ValueError("g* columns contain non-finite values.")

    return Temp_in_GeV, g_star_tab, g_s_tab


def _floor_indices_desc(T_desc: np.ndarray, Tq_array: np.ndarray) -> np.ndarray:
    """
    Return indices of floor values in descending temperature grid.

    Parameters
    ----------
    T_desc : (N,) array-like
        Temperature grid in descending order.
    Tq_array : array-like
        Query temperatures.

    Returns
    -------
    np.ndarray
        Indices i such that T_desc[i] <= Tq.
    """
    T_asc = T_desc[::-1]
    n = T_asc.size
    idx_asc = np.searchsorted(T_asc, Tq_array, side="right") - 1
    idx_asc = np.clip(idx_asc, 0, n - 1)
    return (n - 1) - idx_asc


def g_star(T: np.ndarray | float) -> np.ndarray | float:
    """
    Return g_star(T) from tabulated values using floor match.

    Parameters
    ----------
    T : float or array-like (GeV)

    Returns
    -------
    float or np.ndarray
        g_star(T), matched to largest tabulated T <= query T
    """
    Temp_in_GeV, g_star_tab, _ = load_eff_rel_dof()
    Tq = np.atleast_1d(np.asarray(T, dtype=float))
    idx = _floor_indices_desc(Temp_in_GeV, Tq)
    out = g_star_tab[idx]
    return float(out[0]) if np.isscalar(T) else out


def g_s(T: np.ndarray | float) -> np.ndarray | float:
    """
    Return g_s(T) from tabulated values using floor match.

    Parameters
    ----------
    T : float or array-like (GeV)

    Returns
    -------
    float or np.ndarray
        g_s(T), matched to largest tabulated T <= query T
    """
    Temp_in_GeV, _, g_s_tab = load_eff_rel_dof()
    Tq = np.atleast_1d(np.asarray(T, dtype=float))
    idx = _floor_indices_desc(Temp_in_GeV, Tq)
    out = g_s_tab[idx]
    return float(out[0]) if np.isscalar(T) else out


__all__ = ["load_eff_rel_dof", "g_star", "g_s", "set_custom_eff_rel_dof"]
