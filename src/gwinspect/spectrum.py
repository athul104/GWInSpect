# src/gwinspect/spectrum.py
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Compute the present-day GW spectral energy density Ω_GW(f) across multiple epochs.

Inputs:
    eos_list     : [w_1, w_2, ..., w_n]     (equation-of-state values per epoch; each in [-0.28, 1))
    energy_list  : [E_{n-1}, ..., E_2, E_1] end energies (GeV) from LATEST → EARLIEST epoch
                   (i.e., low → high in energy). Internally we reorder to match w_1..w_n.

API:
    get_omegagw(
        eos_list, energy_list,
        *, E_rstar=None, T_rstar=None,      # beginning of hot Big Bang (≃ end of reheating), GeV
        r=None, E_inf=None,                 # provide exactly one
        num_of_points=1000,
        want_freq_list=False,
        want_efold_list=False
    ) -> (f_arr, Omega_GW_arr[, freq_boundaries][, efold_list])
"""

from __future__ import annotations

import numpy as np
import mpmath
from scipy.special import gamma  # use SciPy per your paper

# package imports
from .bg_constant import m_P, A_S, T_0, Omega_rad_0
from .convert import Temp, vec_Temp, freq, energy_from_T, Num_of_e_folds
from .thermo import g_star_k, g_s_k


def get_omega_gw(
    eos_list,
    energy_list,
    *,
    E_rstar: float | None = None,
    T_rstar: float | None = None,
    r: float | None = None,
    E_inf: float | None = None,
    num_of_points: int = 1000,
    want_freq_list: bool = False,
    want_efold_list: bool = False,
):
    """
    Parameters
    ----------
    eos_list : sequence of floats
        Equation-of-state values [w_1, w_2, ..., w_n] for the reheating epochs. Each must lie in [-0.28, 1).
    energy_list : sequence of floats
        End energy scales (GeV) for the epochs, provided from LATEST → EARLIEST:
        [E_{n-1}, ..., E_2, E_1]. Internally they are reordered to match w_1..w_n.
    E_rstar, T_rstar : float, optional (GeV)
        Beginning of the hot Big Bang phase (≃ end of reheating). Provide exactly one of these.
    r, E_inf : float, optional
        Inflation scale. Provide exactly one: either tensor-to-scalar ratio r (at k*=0.05 Mpc^-1),
        or an energy scale E_inf [GeV].
    num_of_points : int
        Number of log-spaced frequency samples.
    want_freq_list : bool
        If True, also return [f_inf] + transition frequencies used internally.
    want_efold_list : bool
        If True, also return the list of per-epoch e-folds N_e.

    Returns
    -------
    f_arr : (N,) array
        Frequency grid [Hz].
    Omega_GW_arr : (N,) array
        Ω_GW(f) today.
    freq_boundaries : list of floats, optional
        Only if want_freq_list=True: [f_inf] + transition frequencies.
    efold_list : list of floats, optional
        Only if want_efold_list=True: N_e for each reheating epoch.
    """

    # --- Inflation scale from r or E_inf ------------------------------------
    if (r is None) == (E_inf is None):
        raise ValueError("Provide exactly one of (r, E_inf).")
    if r is not None:
        if r >= 0.036:
            raise ValueError("r exceeds the current bound (< 0.036).")
        H_inf = m_P * np.pi * np.sqrt(A_S * r / 2.0)
        E_inf = 3.0 ** 0.25 * np.sqrt(m_P * H_inf)
    else:
        if E_inf >= 1.39e16:
            raise ValueError("E_inf exceeds the current bound (< 1.39e16 GeV).")
        H_inf = E_inf**2 / (np.sqrt(3.0) * m_P)

    # --- Validate EoS values -------------------------------------------------
    lo, hi = (-0.28, 1.0)
    for i, w in enumerate(eos_list):
        if not (lo <= w < hi):
            raise ValueError(f"EoS at position {i+1} must be in [{lo}, {hi}).")

    # --- Reorder energies to match w_1..w_n (earliest → latest) --------------
    # User passes latest→earliest; we sort descending numerically (high→low)
    # so that sorted_energy aligns with w_1..w_n in time order.
    sorted_energy = sorted(energy_list, reverse=True)

    # Range checks for energies
    for i, E_scale in enumerate(sorted_energy, start=1):
        if not (1e-3 <= E_scale <= 1e16):
            raise ValueError(f"Energy scale at position {i} is out of range [1e-3, 1e16] GeV.")
    if sorted_energy and (E_inf < sorted_energy[0]):
        raise ValueError("E_inf is less than the first epoch-end energy. Increase r or E_inf.")

    # --- Reheating end / start of hot Big Bang (T_rstar or E_rstar) ---------
    if (T_rstar is None) == (E_rstar is None):
        raise ValueError("Provide exactly one of (T_rstar, E_rstar), in GeV.")
    if T_rstar is None:
        T_rstar = Temp(E_rstar)

    if T_rstar < 1e-3:
        raise ValueError("T_rstar must be >= 1e-3 GeV.")
    if sorted_energy and (T_rstar >= Temp(sorted_energy[-1])):
        raise ValueError("T_rstar is >= the effective T at the end of the second-last epoch.")

    # --- E-folds per epoch (>= 1) -------------------------------------------
    energy_efolds_check = [E_inf] + sorted_energy + [energy_from_T(T_rstar)]
    efold_list = []
    for i, w in enumerate(eos_list):
        N_e = Num_of_e_folds(w, energy_efolds_check[i], energy_efolds_check[i + 1])
        efold_list.append(float(N_e))
        if N_e < 1.0:
            raise ValueError(f"Epoch {i+1} has N_e < 1. Adjust energies/EoS.")

    # --- Temperatures at transitions ----------------------------------------
    temperature_list = vec_Temp(np.array(sorted_energy)).tolist() if sorted_energy else []
    temperature_list.append(T_rstar)

    if len(temperature_list) != len(eos_list):
        raise ValueError("Number of epochs (eos_list) must match number of transition energies/temperatures.")

    # Append RD and MD stages
    eos_full = list(eos_list) + [1.0/3.0, 0.0]
    temperature_list.append(1e-9)  # T_eq placeholder (as in your original)

    # Frequencies at transitions
    freq_list = [freq(T) for T in temperature_list]
    alpha_arr = 2.0 / (1.0 + 3.0 * np.array(eos_full, dtype=float))

    # --- Matching coefficients across epochs --------------------------------
    def coeff(f):
        """Return (A_k_n, B_k_n) for the final epoch at frequency f."""
        A_k_arr = np.zeros(len(alpha_arr))
        B_k_arr = np.zeros(len(alpha_arr))
        y_arr = np.zeros(len(alpha_arr) - 1)

        for i in range(len(y_arr)):
            y_arr[i] = f / freq_list[i]

        A_k_arr[0] = 2 ** (alpha_arr[0] - 0.5) * gamma(alpha_arr[0] + 0.5)  # Eq. (2.26)
        B_k_arr[0] = 0.0

        for i in range(1, len(alpha_arr)):
            an_ym = alpha_arr[i] * y_arr[i - 1]
            am_ym = alpha_arr[i - 1] * y_arr[i - 1]

            an_m_half = alpha_arr[i] - 0.5
            am_m_half = alpha_arr[i - 1] - 0.5
            an_p_half = alpha_arr[i] + 0.5
            am_p_half = alpha_arr[i - 1] + 0.5

            C = (an_ym ** (an_m_half)) / (am_ym ** (am_m_half))  # prefactor

            f_1 = mpmath.besselj(-(an_m_half), an_ym)
            f_2 = mpmath.besselj(-(am_m_half), am_ym)
            f_3 = mpmath.besselj(-(an_p_half), an_ym)
            f_4 = mpmath.besselj(-(am_p_half), am_ym)

            g_1 = mpmath.besselj(an_m_half, an_ym)
            g_2 = mpmath.besselj(am_m_half, am_ym)
            g_3 = mpmath.besselj(an_p_half, an_ym)
            g_4 = mpmath.besselj(am_p_half, am_ym)

            Deno = f_1 * g_3 + g_1 * f_3
            K = C / Deno

            Num_A1 = g_2 * f_3 + g_4 * f_1
            Num_B1 = f_2 * f_3 - f_4 * f_1

            Num_A2 = g_2 * g_3 - g_4 * g_1
            Num_B2 = f_2 * g_3 + f_4 * g_1

            A_prev, B_prev = A_k_arr[i - 1], B_k_arr[i - 1]
            A_k_arr[i] = K * (A_prev * Num_A1 + B_prev * Num_B1)
            B_k_arr[i] = K * (A_prev * Num_A2 + B_prev * Num_B2)

        return A_k_arr[-1], B_k_arr[-1]

    # --- Relativistic correction at start of last RD epoch -------------------
    G_R = (g_star_k(temperature_list[-2]) / g_star_k(T_0)) * (g_s_k(T_0) / g_s_k(temperature_list[-2])) ** (4.0 / 3.0)
    const_coeff = (1.0 / (96.0 * (np.pi) ** 3)) * G_R * Omega_rad_0 * (H_inf / m_P) ** 2

    # --- Present-day Ω_GW(f) -------------------------------------------------
    def Omega_GW_0(f):
        y_eq = f / freq_list[-1]  # f/f_eq
        A_k, B_k = coeff(f)
        return const_coeff * y_eq ** (-2.0) * (A_k**2 + B_k**2)

    vec_Omega_GW_0 = np.vectorize(Omega_GW_0)

    # Frequency grid
    f_inf = freq(Temp(E_inf))                          # present frequency for inflation scale
    start_freq = np.log10(2.0e-20)                     # Hz
    end_freq = np.log10(f_inf)                         # Hz
    f_arr = np.logspace(start_freq, end_freq, num_of_points, endpoint=True, base=10.0)

    Omega_GW_arr = vec_Omega_GW_0(f_arr)

    # Optional extras
    out = [f_arr, Omega_GW_arr]
    if want_freq_list:
        freq_boundaries = [f_inf] + freq_list
        out.append(freq_boundaries)
    if want_efold_list:
        out.append(efold_list)

    return tuple(out)
