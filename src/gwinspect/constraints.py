# src/gwinspect/constraint.py
# SPDX-License-Identifier: GPL-3.0-or-later
'''Module to evaluate Big Bang Nucleosynthesis (BBN) constraints on gravitational wave energy density
in multi-epoch pre-hot Big Bang cosmological scenarios.

Inputs:
- Equation-of-state parameters for each epoch.
- Energy scales marking transitions between epochs.
- Inflation scale (tensor-to-scalar ratio or energy scale).
- Temperature or energy scale at the end of pre-hot Big Bang phase.

Outputs:
- Computed left-hand side value of the BBN constraint piecewise integral.
- Boolean indicating if the BBN constraint is satisfied (LHS < bbn_bound).
'''


from __future__ import annotations

import numpy as np

from .constants import m_P, A_S, T0, omega_rad0, T_bbn
from .cosmo_tools import temp_of_E, freq_of_T, energy_of_T, compute_efolds
from .thermo import g_star, g_s




def check_bbn(
    eos_list: list[float],
    energy_list: list[float],
    *,
    E_rstar: float | None = None,
    T_rstar: float | None = None,
    r: float | None = None,
    E_inf: float | None = None,
    bbn_bound: float = 1.13e-6,
    tol: float = 1e-12,
):
    """Check if a given multi-epoch pre-hot Big Bang scenario satisfies the BBN constraint on the
    gravitational wave energy density.

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
    bbn_bound : float
        Upper bound on the gravitational wave energy density from BBN constraints. Default is 1.13e-6.
    tol : float
        Tolerance value for numerical stability in calculations. If abs(alpha - 1.0) < tol during any epoch, logarithmic limit is used.
        Default is 1e-12.

    Returns
    -------
    piecewise integral : float
        Computed left-hand side value of the BBN constraint piecewise integral.
    if_satisfied : bool
        True if the BBN constraint is satisfied (LHS < bbn_bound), False otherwise.
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
    sorted_energy = sorted(energy_list, reverse=True)
    for i in range(1, len(sorted_energy)):
        if sorted_energy[i] >= sorted_energy[i - 1]:
            raise ValueError("energy_list provided is not strictly increasing from latest to earliest epoch.")

    if T_rstar >= temp_of_E(sorted_energy[-1]):
        raise ValueError("T_rstar >= the temperature corresponding to the beginning of the final pre-hot Big Bang epoch. Adjust energies or T_rstar.")
    if sorted_energy[0] >= E_inf:
        raise ValueError("The energy corresponding to the end of the first pre-hot Big Bang epoch >= E_inf. Adjust energies or E_inf or r.")

    # --- E-folds per epoch ---
    energy_boundaries = [E_inf] + sorted_energy + [energy_of_T(T_rstar)]
    for i, w in enumerate(eos_list):
        N_e = compute_efolds(w=w, Ei=energy_boundaries[i], Ef=energy_boundaries[i + 1])
        if N_e < 1.0:
            raise ValueError(f"Epoch {i+1} has N_e < 1. Adjust energies or EoS.")


    # --- Temperature and EoS list ---
    T_list = temp_of_E(np.array(energy_boundaries))  # [T_inf, T1, T2, ..., Tn = T_rstar]
    freq_list = [freq_of_T(T) for T in T_list] # [f_inf, f1, f2, ..., fn = f_rstar]
    alpha_arr = 2.0 / (1.0 + 3.0 * np.array(eos_list))  # [α1, α2, ..., αn]
    f_bbn  = freq_of_T(T_bbn) # BBN frequency [Hz]

    omega_gw_rad0 = 1/12 * (H_inf / (np.pi * m_P))**2 * omega_rad0 # h^2 \Omega_gw during radiation era

    cal_F_arr = [1] * len(eos_list)  # F_i factors
    for i in range(len(alpha_arr) - 1):
        for j in range(i + 1, len(eos_list)):
            cal_F_arr[i] *= (freq_list[j] / freq_list[j + 1])**(2 * (1 - alpha_arr[j]))

    # --- LHS integral ---
    lhs_value = 0.0
    for i, alpha in enumerate(alpha_arr):
        f_in = freq_list[i]
        f_fi   = freq_list[i + 1]
        cal_F   = cal_F_arr[i]

        if abs(alpha - 1.0) < tol:
            # log-limit
            integral_piece = cal_F * np.log(f_in / f_fi)
        else:
            integral_piece = cal_F / (2 * (1 - alpha)) * ((f_in / f_fi)**(2 * (1 - alpha)) - 1.0)

        lhs_value +=  integral_piece
    
    lhs_value += np.log(freq_list[-1] / f_bbn)  # final radiation-dominated part
    lhs_value *= omega_gw_rad0
    rhs_bound = bbn_bound
    
    return_values = (lhs_value, lhs_value < rhs_bound)

    return return_values



__all__ = ["check_bbn"]
