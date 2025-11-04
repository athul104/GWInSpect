# src/gwinspect/constraint.py
# SPDX-License-Identifier: GPL-3.0-or-later
"""
BBN piecewise constraint (Eq. 2.21–2.23 in the companion paper).

API
---
check_bbn(
    eos_list, energy_list,
    *, E_rstar=None, T_rstar=None,        # start of hot Big Bang (≃ end of reheating), GeV
    r=None, E_inf=None,                   # provide exactly one
    bbn_bound = 1.13e-6                   # numeric prefactor in RHS of Eq. (2.23)
    t_bbn_GeV: float | None = 1.0e-3,     # default T_bbn ≈ 1 MeV, used if f_bbn not given
    f_bbn: float | None = None,           # alternatively pass f_BBN directly [Hz]
    eps: float = 1e-12,                   # tolerance for handling w≈1/3 (α≈1)
    want_details: bool = False
) -> tuple
    Returns (lhs_value, rhs_bound, passes) and optionally a details dict.

Conventions
-----------
- eos_list = [w1, w2, ..., wn]  (earliest → latest reheating epochs)
- energy_list is provided as [E_{n-1}, ..., E_2, E_1] (LATEST → EARLIEST).
  Internally we reorder to match w1..wn (high → low energy).
"""

from __future__ import annotations

import numpy as np

from .constants import m_P, A_S, T0, omega_rad0
from .cosmo_tools import temp_of_E, freq_of_T, energy_of_T
from .thermo import g_star, g_s


def _alpha_from_w(w: float) -> float:
    """α = 2/(1+3w)."""
    return 2.0 / (1.0 + 3.0 * w)


def check_bbn(
    eos_list,
    energy_list,
    *,
    E_rstar: float | None = None,
    T_rstar: float | None = None,
    r: float | None = None,
    E_inf: float | None = None,
    t_bbn_GeV: float | None = 1.0e-3,
    f_bbn: float | None = None,
    bbn_bound: float = 1.13e-6,
    eps: float = 1e-12,
    want_details: bool = False,
):
    """
    Compute the LHS of the BBN inequality (piecewise integral) and the RHS bound.

    Parameters
    ----------
    eos_list : sequence of floats
        [w1, w2, ..., wn] for reheating epochs (each in [-0.28, 1)).
    energy_list : sequence of floats
        End energies (GeV) provided as [E_{n-1}, ..., E_2, E_1] (LATEST → EARLIEST).
        Internally reordered (high → low) to align with w1..wn.
    E_rstar, T_rstar : float, optional (GeV)
        Beginning of the hot Big Bang (≃ end of reheating). Provide exactly one.
    r, E_inf : float, optional
        Inflation scale: provide exactly one (tensor-to-scalar ratio r, or energy E_inf [GeV]).
    t_bbn_GeV : float, optional
        Temperature used to set f_BBN if f_bbn is None (default 1 MeV).
    f_bbn : float, optional
        If given, overrides t_bbn_GeV and uses this as f_BBN [Hz].
    eps : float
        Tolerance for treating w≈1/3 (α≈1) with the log-limit.
    want_details : bool
        If True, also return a dict with intermediate arrays for debugging.

    Returns
    -------
    lhs_value : float
        Value of the left-hand side in Eq. (2.21).
    rhs_bound : float
        Value of the right-hand side (1.13e-6 × (h^2 Ω_GW^0,RD)^(-1)).
    passes : bool
        True iff lhs_value < rhs_bound.
    details : dict, optional
        Returned only if want_details=True (contains alphas, F_i, f-list, etc.).
    """
    # ---- inflation scale from (r or E_inf) ---------------------------------
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

    # ---- reorder energies latest→earliest  → earliest→latest ----------------
    sorted_energy = sorted(energy_list, reverse=True)  # high → low, matches w1..wn
    n = len(eos_list)
    if len(sorted_energy) != n:
        raise ValueError("eos_list and energy_list must have the same length.")

    # ---- T_r* from (T_rstar or E_rstar) ------------------------------------
    if (T_rstar is None) == (E_rstar is None):
        raise ValueError("Provide exactly one of (T_rstar, E_rstar), in GeV.")
    if T_rstar is None:
        T_rstar = temp_of_E(E_rstar)
    if T_rstar < 1e-3:
        raise ValueError("T_rstar must be >= 1e-3 GeV.")
    if sorted_energy and (T_rstar >= temp_of_E(sorted_energy[-1])):
        raise ValueError("T_rstar is >= the effective T at the end of the second-last epoch.")

    # ---- transition temperatures & frequencies ------------------------------
    T_trans = np.asarray(temp_of_E(np.array(sorted_energy)), dtype=float)  # [T1, T2, ..., Tn]
    f_trans = np.asarray([freq_of_T(T) for T in T_trans], dtype=float)         # [f1, f2, ..., fn]
    f_rstar = freq_of_T(T_rstar)                                               # should equal fn
    # safety: replace last with exact f_rstar (keeps order)
    if n >= 1:
        f_trans[-1] = f_rstar

    # f_end (end of inflation) and f_BBN
    f_end = freq_of_T(temp_of_E(E_inf))
    if f_bbn is None:
        if t_bbn_GeV is None:
            raise ValueError("Either set f_bbn or t_bbn_GeV.")
        f_bbn = float(freq_of_T(t_bbn_GeV))

    # ---- α_i and cumulative ℱ_i factors (Eq. 2.22) -------------------------
    alphas = np.array([_alpha_from_w(w) for w in eos_list], dtype=float)  # [α1..αn]

    # ℱ_n = 1. For i<n: ℱ_i = (f_i/f_{i+1})^{2(1-α_{i+1})} * ℱ_{i+1}
    F = np.ones(n, dtype=float)
    for i in range(n - 2, -1, -1):
        alpha_next = alphas[i + 1]
        F[i] = F[i + 1] * (f_trans[i] / f_trans[i + 1]) ** (2.0 * (1.0 - alpha_next))

    # ---- LHS of Eq. (2.21) --------------------------------------------------
    lhs = np.log(f_rstar / f_bbn)  # RD piece (α=1)

    # sum terms: i = n .. 1  (our index j = i-1 from n-1 down to 0)
    for j in range(n - 1, -1, -1):
        alpha_i = alphas[j]
        f_lower = f_trans[j]
        f_upper = f_end if j == 0 else f_trans[j - 1]  # f_{i-1}
        if abs(1.0 - alpha_i) < eps:
            # limit α_i → 1  ⇒  F_i * ln(f_{i-1}/f_i)
            term = F[j] * np.log(f_upper / f_lower)
        else:
            expo = 2.0 * (1.0 - alpha_i)
            term = F[j] / (2.0 * (1.0 - alpha_i)) * ((f_upper / f_lower) ** expo - 1.0)
        lhs += term

    # ---- RHS bound (Eq. 2.23 + numeric prefactor) ---------------------------
    # h^2 Ω_GW^0,RD ≃ (1/24) Ω_rad,0 × (2/π^2) × (H_inf/m_P)^2
    h2_OmegaGW0_RD = (1.0 / 24.0) * omega_rad0 * (2.0 / (np.pi**2)) * (H_inf / m_P) ** 2
    rhs = bbn_bound * (h2_OmegaGW0_RD ** -1)

    passes = bool(lhs < rhs)

    if want_details:
        details = dict(
            alphas=alphas,
            F=F,
            T_trans=T_trans,
            f_trans=f_trans,
            f_end=f_end,
            f_rstar=f_rstar,
            f_bbn=f_bbn,
            H_inf=H_inf,
            h2_OmegaGW0_RD=h2_OmegaGW0_RD,
        )
        return float(lhs), float(rhs), passes, details
    return float(lhs), float(rhs), passes


__all__ = ["check_bbn"]
