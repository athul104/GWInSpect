# src/gwinspect/spectrum.py
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Compute the present-day spectral energy density Ω_GW(f) of inflationary first-order gravitational waves
resulting from an arbitrary sequence of pre-hot Big Bang post-inflationary epochs.

Inputs:
    eos_list     : [w_1, w_2, ..., w_n]     (equation-of-state values per epoch; each in [-0.28, 1))
    energy_list  : [E_{n-1}, ..., E_2, E_1] end energies (GeV) from LATEST → EARLIEST epoch
                   (i.e., lowest to highest in energy). Internally reordered to align with w_1..w_n.

Main API:
    compute_omega_gw(
        eos_list, energy_list,
        *, E_rstar=None, T_rstar=None,      # beginning of hot Big Bang (end of reheating), GeV
        r=None, E_inf=None,                 # inflationary scale: provide exactly one
        num_of_points=1000,
        f_min=2e-20, f_max=None,
        show_freqs=False,
        show_efolds=False
    ) -> (f_arr, omega_gw_arr[, freq_boundaries][, efold_list])
"""

from __future__ import annotations

from mpmath.libmp.libelefun import k
import numpy as np
from mpmath import besselj
from scipy.special import gamma

from typing import Union, Sequence


# Package-level constants and imports
from .constants import T_eq, m_P, A_S, T0, omega_rad0, T_bbn
from .cosmo_tools import temp_of_E, freq_of_T, energy_of_T, compute_efolds
from .thermo import g_star, g_s

# Define the type for f_custom argument (float, list of floats, or np.ndarray)
ArrayLike = Union[float, Sequence[float], np.ndarray]

def _prepare_frequency_array(
    f_min: float,
    f_max: float | None,
    f_inf: float,
    num_of_points: int,
    f_custom: ArrayLike | None = None,
) -> np.ndarray:
    """
    Helper function to prepare the frequency grid for Ω_GW computation.

    Parameters
    ----------
    f_min : float
        Minimum frequency for logspace grid [Hz].
    f_max : float or None
        Maximum frequency for logspace grid [Hz]. Defaults to f_inf.
    f_inf : float
        Inflationary frequency cutoff [Hz].
    num_of_points : int
        Number of log-spaced points (if f_custom not given).
    f_custom : array-like, optional
        If provided, use this custom frequency or frequencies instead.

    Returns
    -------
    np.ndarray
        Frequency array for Ω_GW computation.
    """
    if f_custom is not None:
        for f in f_custom:
            if f <= 0 or not np.isfinite(f):
                raise ValueError("All custom frequencies must be positive and finite.")
            if f > f_inf:
                raise ValueError("All custom frequencies must be <= f_inf.")
            if f < 2e-20:
                print("Warning: custom frequency below 2e-20 Hz may correspond to super-horizon modes today.")
        f_custom = np.asarray(f_custom, dtype=float)
        return f_custom
    
    if f_min < 2e-20:
        print("Warning: f_min below 2e-20 Hz may correspond to super-horizon modes today.")

    if f_max is not None:
        if f_max > f_inf:
            raise ValueError("f_max must be <= f_inf.")
        end_logf = np.log10(f_max)
    else:
        end_logf = np.log10(f_inf)
    
    start_logf = np.log10(f_min)
    return np.logspace(start_logf, end_logf, num=num_of_points, endpoint=True, base=10.0)

def compute_omega_gw(
    eos_list: list[float],
    energy_list: list[float],
    *,
    E_rstar: float | None = None,
    T_rstar: float | None = None,
    r: float | None = None,
    E_inf: float | None = None,
    f_custom: float | None = None,
    f_min: float = 2e-20,
    f_max: float | None = None,
    num_of_points: int = 1000,
    show_freqs: bool = False,
    show_efolds: bool = False,
    k_pivot: float = 0.05,
    n_T = 0,
):
    """
    Compute the present-day spectral energy density Ω_GW(f) of inflationary first-order gravitational waves
    resulting from an arbitrary sequence of pre-hot Big Bang post-inflationary epochs.

    Parameters
    ----------
    eos_list : list of float
        Equation-of-state values [w_1, ..., w_n] for pre-hot Big Bang epochs. Each w ∈ [-0.28, 1).
    energy_list : list of float
        List of end energies [E_{n-1}, ..., E_2, E_1] (in GeV), provided from latest to earliest pre-hot Big Bang epoch.
        This list does not include the energy scale at the end of pre-hot Big Bang (E_rstar).
        If pre-hot Big Bang has n epochs, energy_list must have length n-1.
        For single equation-of-state epoch (n=1), provide an empty list [].
    E_rstar, T_rstar : float
        Energy scale (E_rstar) or temperature (T_rstar) in GeV marking the beginning of the hot Big Bang (≃ end of reheating).
        You must specify exactly one of these. If both are provided, T_rstar is used.
    r, E_inf : float
        Inflation scale: tensor-to-scalar ratio (r) at k*=0.05 Mpc^-1 or energy scale during inflation (E_inf) in GeV.
        You must specify exactly one of these. If both are provided, r is used.
    f_custom : list or np.ndarray, optional
        If provided, compute Ω_GW only at these frequency/frequencies [Hz].
    f_min, f_max : float, optional
        Frequency limits in Hz for the output spectrum.
        If f_max is None, use the value corresponding to the end of inflation, f_inf computed internally.
        If f_min < 2e-20 Hz, a warning is printed since this may correspond to super-horizon modes today.
    num_of_points : int
        Number of frequency samples. Default is 1000.
    show_freqs : bool
        If True, return the list of transition frequencies for the pre-hot Big Bang phase, [f_0 = f_inf, f_1, ..., f_n = f_{r*}].
    show_efolds : bool
        If True, return the list of e-folds per epoch in the pre-hot Big Bang phase, [N_e1, N_e2, ..., N_en].
        The function also internally checks that each epoch has N_e >= 1. If not, a ValueError is raised.
    k_pivot : float
        CMB Pivot scale in Mpc^-1. Default is 0.05.
    n_T : float
        inflationary tensor spectral index. Default is 0.

    Returns
    -------
    tuple
        Depending on show_freqs and show_efolds, returns:
        - (f_arr, omega_gw_arr, freq_boundaries (if show_freqs), efold_list (if show_efolds))

        where:
            f_arr : np.ndarray
                Frequency array [Hz].
            omega_gw_arr : np.ndarray
                Corresponding present-day spectral energy density Ω_GW(f).
            freq_boundaries : list of float, optional
                If show_freqs=True, list of transition frequencies during pre-hot Big Bang phase [f_0 = f_inf, f_1, ..., f_n = f_{r*}] in Hz.
            efold_list : list of float, optional
                If show_efolds=True, list of e-folds per pre-hot Big Bang epoch [N_e1, N_e2, ..., N_en].
    """

    if len(eos_list) != len(energy_list) + 1:
        raise ValueError("Length mismatch: len(energy_list) must be len(eos_list) - 1.")

    # --- Inflation scale: H_inf, E_inf ---
    if (r is None) == (E_inf is None):
        raise ValueError("Provide exactly one of (r, E_inf).")
    if r is not None:
        if r >= 0.036:
            raise ValueError("r exceeds the current bound (< 0.036).")
        H_inf = m_P * np.pi * np.sqrt(A_S * r / 2.0)
        E_inf = (3.0**0.25) * np.sqrt(m_P * H_inf)
    else:
        if E_inf >= 1.4e16:
            raise ValueError("E_inf exceeds the current bound (< 1.4e16 GeV), corresponding to r < 0.036).")
        H_inf = E_inf**2 / (np.sqrt(3.0) * m_P)

    # --- EoS validation ---
    for i, w in enumerate(eos_list):
        if not (-0.28 <= w < 1.0):
            raise ValueError(f"Equation of state, w, provided in eos_list at position {i+1} is out of bounds [-0.28, 1).")

    # --- Reheating end: T_rstar ---
    if (T_rstar is None) == (E_rstar is None):
        raise ValueError("Provide exactly one of (T_rstar, E_rstar).")
    if T_rstar is None:
        T_rstar = temp_of_E(E_rstar)
    if T_rstar < T_bbn:
        raise ValueError(f"The temperature corresponding to the end of pre-hot Big Bang phase, T_rstar, is below BBN temperature ({T_bbn} GeV).")

    # --- Reorder energies: latest → earliest input → time-ordered (high → low) ---
    if energy_list:
        sorted_energy = sorted(energy_list, reverse=True)
        for i in range(1, len(sorted_energy)):
            if sorted_energy[i] >= sorted_energy[i - 1]:
                raise ValueError("energy_list provided is not strictly increasing from latest to earliest epoch.")

        if T_rstar >= temp_of_E(sorted_energy[-1]):
            raise ValueError("T_rstar >= the temperature corresponding to the beginning of the final pre-hot Big Bang epoch. Adjust energies or T_rstar.")
        if sorted_energy[0] >= E_inf:
            raise ValueError("The energy corresponding to the end of the first pre-hot Big Bang epoch >= E_inf. Adjust energies or E_inf or r.")
    else:
        sorted_energy = []
    # --- E-folds per epoch ---
    energy_boundaries = [E_inf] + sorted_energy + [energy_of_T(T_rstar)]
    efold_list = []
    for i, w in enumerate(eos_list):
        N_e = compute_efolds(w=w, Ei=energy_boundaries[i], Ef=energy_boundaries[i + 1])
        if N_e < 1.0:
            raise ValueError(f"Epoch {i+1} has N_e < 1. Adjust energies or EoS.")
        efold_list.append(float(N_e))

    # --- Temperature and EoS lists including RD, MD ---
    T_list = temp_of_E(np.array(sorted_energy)).tolist() + [T_rstar, T_eq]
    w_full = list(eos_list) + [1/3, 0]
    freq_list = [freq_of_T(T) for T in T_list]
    alpha_arr = 2.0 / (1.0 + 3.0 * np.array(w_full))

    # --- Coefficient propagation function ---
    def _coeff(f: float) -> tuple[float, float]:
        '''
        Compute the Bogoliubov coefficients (A_k, B_k) in mode function at frequency f
        by propagating through all epochs using Israel junction conditions.
        
        Parameters
        ----------
        f : float
            Frequency in Hz.

        Returns
        -------
        tuple[float, float]
            Bogoliubov coefficients (A_k, B_k) in the mode function at frequency f for the final matter-dominated epoch in the hot Big Bang phase.
        '''
        len_alpha = len(alpha_arr)
        A_k, B_k = [0.0] * len_alpha, [0.0] * len_alpha
        y_arr = f / np.array(freq_list)

        A_k[0] = 2 ** (alpha_arr[0] - 0.5) * gamma(alpha_arr[0] + 0.5)

        for i in range(1, len_alpha):
            a_n, a_m = alpha_arr[i], alpha_arr[i - 1]
            y = y_arr[i - 1]

            an_ym, am_ym = a_n * y, a_m * y

            an_m_half = a_n - 0.5
            am_m_half = a_m - 0.5

            an_p_half = a_n + 0.5
            am_p_half = a_m + 0.5


            C = (an_ym ** an_m_half)/(am_ym ** am_m_half)


            f1 = besselj(-an_m_half, an_ym)
            f2 = besselj(-am_m_half, am_ym)
            f3 = besselj(-an_p_half, an_ym)
            f4 = besselj(-am_p_half, am_ym)

            g1 = besselj(an_m_half, an_ym)
            g2 = besselj(am_m_half, am_ym)
            g3 = besselj(an_p_half, an_ym)
            g4 = besselj(am_p_half, am_ym)

            D = f1 * g3 + g1 * f3
            K = C / D

            num_A1 = g2 * f3 + g4 * f1
            num_B1 = f2 * f3 - f4 * f1

            num_A2 = g2 * g3 - g4 * g1
            num_B2 = f2 * g3 + f4 * g1

            A_k[i] = K * (A_k[i-1] * num_A1 + B_k[i-1] * num_B1)
            B_k[i] = K * (A_k[i-1] * num_A2 + B_k[i-1] * num_B2)

        return float(A_k[-1]), float(B_k[-1])

    # --- Normalization prefactor ---
    G_R = (g_star(T_rstar) / g_star(T0)) * (g_s(T0) / g_s(T_rstar)) ** (4/3)
    norm = (1.0 / (96 * np.pi**3)) * G_R * omega_rad0 * (H_inf / m_P) ** 2

    def _omega_gw(f: float) -> float:
        '''Compute Ω_GW(f) at present day for frequency f (Hz).
        
        Parameters
        ----------
        f : float
            Frequency in Hz.
        
        Returns
        -------
        float
            Present-day spectral energy density Ω_GW(f).
        '''

        f_pivot = (9.70e-15 * k_pivot)/(2*np.pi) #Hz, CMB Pivot frequency at present day

        A, B = _coeff(f)
        return norm * (f / freq_list[-1])**(-2) * (A**2 + B**2) * (f / f_pivot)**n_T

    f_inf = freq_of_T(temp_of_E(E_inf))

    f_arr = _prepare_frequency_array(f_min, f_max, f_inf, num_of_points, f_custom)
    omega_gw_arr = np.array([_omega_gw(f) for f in f_arr], dtype=float)

    out = [f_arr, omega_gw_arr]
    if show_freqs:
        out.append([f_inf] + freq_list[:-1])
    if show_efolds:
        out.append(efold_list)

    return tuple(out)
