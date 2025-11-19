# GWInSpect

**Gravitational Wave Inflationary Spectra**

[![PyPI version](https://img.shields.io/pypi/v/gwinspect?color=blue)](https://pypi.org/project/gwinspect)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-brightgreen)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)

---

**GWInSpect** is a lightweight, research-grade Python package that computes:

- The **present-epoch spectral energy density** of first-order inflationary gravitational waves, $\Omega_{\rm GW}(f) $, produced by a given (user-defined) post-inflationary expansion history.
- Verifies whether the **BBN constraint** on (integrated) gravitational wave energy density is satisfied.

---

## 📦 Installation

Install ```GWInSpect``` by running any one of the following commands in your terminal. 

### 1. ✅ From PyPI (Recommended)

```bash
pip install gwinspect
```

### 2. 🧪 From GitHub

```bash
pip install git+https://github.com/athul104/GWInSpect.git
```

### 3. 🛠️ Cloning repo (to make an editable copy of the package in your local system)

```bash
git clone https://github.com/athul104/GWInSpect.git
cd GWInSpect
pip install -e .
```

## 📋 Requirements

The following necessary libraries will be automatically installed when you install ```GWInSpect```.

* Python **3.9** or later
* ```numpy >= 1.21```
* ```scipy >= 1.9```
* ```mpmath >= 1.2.0```

## 📓 Tutorial Notebook

The package includes a tutorial notebook under [`examples/tutorial_gwinspect.ipynb`](examples/tutorial_gwinspect.ipynb)

## 🧱 Package Structure

Each module is independently documented and demonstrated in the tutorial notebook.

* ```spectrum.py```
    * ```compute_omega_gw```  
Computes $\Omega_{\rm GW}(f)$ for a user defined post-inflationary history of the Universe consisting of a sequence of pre-hot Big Bang epochs, each with a constant equation of state. 

* ```constraints.py```
    * ```check_bbn```  
Computes BBN constraint integral piecewise for the user defined sequence of post-inflationary epochs.

* ```cosmo_tools.py```
Utilities for cosmological parameter conversions:
    * ```temp_of_E```
Convert temperature to energy scale
    * ```energy_of_T```
Convert energy scale to temperature
    * ```freq_of_T```
Convert temperature to present-day frequency of gravitational waves
    * ```compute_efolds```
Compute the duration of each epoch in terms of number of $e$-folds

* ```thermo.py```
    * Functions to compute $g_{*}(T),  g_{s}(T)$ from fitting functions given in **K. Saikawa and S. Shirai** [arXiv:1803.01038](https://arxiv.org/abs/1803.01038).
    * Key functions: ```g_star``` and ```g_s```
    * Users can also load the pre-computed data of [temperature $T$, $g_{*}(T)$, $g_{s}(T)$, energy($T$)] stored in ```src\gwinspect\data\eff_rel_dof.txt``` using ```load_eff_rel_dof()```

* ```constants.py```
    * Contains constants: reduced Planck mass ```m_P```, present CMB temperature ```T0```, scalar amplitude ```A_S```, Present radiation density parameter ```omega_rad0```, BBN temperature ```T_bbn```, matter-radiation equality temperature ```T_eq```
    * Users can also redefine the values of this conatant using ```set_constants```


## 📚 Citation

This package accompanies the paper:

> **Swagat S. Mishra & Athul K. Soman (2025)**  
> *Morphological Zoo of Inflationary Gravitational Wave Spectra imprinted by a Sequence of Post-Inflationary Epochs*  
> [arXiv:2510.25672](https://arxiv.org/abs/2510.25672)

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
©️ GWInSpect authors (2025)


## 🙋 Feedback

Thank you for your interest in **GWInSpect** 💜.  
Please report issues or suggestions at:
[github.com/athul104/GWInSpect/issues](https://github.com/athul104/GWInSpect/issues)

---
