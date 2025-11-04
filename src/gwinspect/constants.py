# src/gwinspect/constants.py

"""
Constants used throughout GWInSpect.

All values are in natural units (GeV-based) unless noted.
Users may override them at runtime using `set_constants(...)`.
"""

__all__ = [
    "omega_rad0", "T0", "m_P", "A_S",
    "T_bbn", "T_eq", "set_constants"
]

# --- Default values (in GeV units etc.) ---
omega_rad0   = 4.16e-5      # Present radiation density parameter (Ω_rad,0 * h^2)
T0           = 2.35e-13     # Present temperature (GeV)
m_P           = 2.44e18      # Reduced Planck mass (GeV)
A_S           = 2.1e-9       # Scalar power amplitude at pivot scale
T_bbn         = 1e-3         # Approx. BBN threshold temperature (GeV)
T_eq          = 1e-9         # Matter-radiation equality temp. (GeV)

def set_constants(**kwargs):
    """
    Update default constants dynamically.
    
    Example:
        set_constants(m_P=2.435e18, A_S=2.105e-9)
    """
    globals().update({k: v for k, v in kwargs.items() if k in globals()})
