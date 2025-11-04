# GWInSpect

**Gravitational Wave Inflationary Spectra**

[![PyPI version](https://img.shields.io/pypi/v/gwinspect?color=blue)](https://pypi.org/project/gwinspect)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-brightgreen)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)

---

**GWInSpect** is a lightweight, research-grade Python package that computes:

- The **present-day spectral energy density** of first-order inflationary gravitational waves \( \Omega_{\rm GW}(f) \), shaped by a user-defined post-inflationary expansion history.
- Whether the **BBN constraint** on total gravitational wave energy density is satisfied.

A full tutorial notebook is included under [`examples/tutorial_gwinspect.ipynb`](examples/tutorial_gwinspect.ipynb).

---

## 📦 Installation

### ✅ From PyPI (Recommended)

```bash
pip install gwinspect
```

### 🧪 From GitHub

```bash
pip install git+https://github.com/athul104/GWInSpect.git
```

### 🛠️ Cloning repo to your local system

```bash
git clone https://github.com/athul104/GWInSpect.git
cd GWInSpect
pip install -e .
```

## 📋 Requirements
* Python **3.9** or later
* ```numpy >= 1.21```
* ```scipy >= 1.9```
* ```mpmath >= 1.2.0```


## 🧱 Package Structure

Each module is independently documented and demonstrated in the tutorial notebook.

* ```spectrum.py```
* * ```compute_omega_gw```  
Computes $\Omega_{\rm GW}(f)$ for user defined post-inflationary scenario consisting of multiple epochs during pre-hot Big Bang phase with different constant equations of state. 

```constraints.py```
* ```check_bbn```  
Computes BBN constraint integral piecewise for the user defined post-inflationary scenario.

```cosmo_tools.py```
Utilities for cosmological conversions:

* ```temp_of_E```
Convert temperature to energy scale
* ```energy_of_T```
Convert energy scale to temperature
* ```freq_of_T```
Convert temperature to present-day frequency of gravitational waves
* ```compute_efolds```
Compute number of $e$-folds extended during an epoch

```thermo.py```
* Uses tabulated $g_{*}(T), \, g_{s}(T)$ from ```data/eff_rel_dof.txt```
* Key functions: ```g_star``` and ```g_s```
* Users can load the data used to compute $g_{*}(T), \, g_{s}(T)$ using ```load_eff_rel_dof```
* Users can also use their own preferred data for relativistic degrees of freedom using ```set_custom_eff_rel_dof```

```constants.py```
* Contains constants: reduced Planck mass ```m_P```, present CMB temperature ```T0```, scalar amplitude ```A_S```, Present radiation density parameter ```omega_rad0```, BBN temperature ```T_bbn```, matter-radiation equality temperature ```T_eq```
* Users can also redefine the values of this conatant using ```set_constants```


## 📓 Tutorial Notebook

The package includes a tutorial notebook:

```bash
examples/tutorial_gwinspect.ipynb
```

This package accompanies the paper:

> **Swagat S. Mishra & Athul K. Soman (2025)**  
> *Morphological Zoo of Inflationary Gravitational Wave Spectra imprinted by a Sequence of Post-Inflationary Epochs*  
> [arXiv:2510.25672](https://arxiv.org/abs/2510.25672)


## 📚 Citation

If you use GWInSpect in academic work, please cite:

```bibtex
@article{Mishra:2025nnu,
    author = "Mishra, Swagat S. and Soman, Athul K.",
    title = "{Morphological Zoo of Inflationary Gravitational Wave Spectra imprinted by a Sequence of Post-Inflationary Epochs}",
    eprint = "2510.25672",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "10",
    year = "2025"
}
```
---

## 🛠 License
GPL-3.0-or-later
© GWInSpect authors (2025)


## 🙋 Feedback

Thank you for your interest in **GWInSpect**.  
Please report issues or suggestions at:
[github.com/athul104/GWInSpect/issues](https://github.com/athul104/GWInSpect/issues)

---
