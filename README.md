# GWInSpect

**GWInSpect** (Gravitational-Wave Inflation Spectrum) is a lightweight, single-file Python module for computing and visualizing the **spectral energy density of primordial gravitational waves** generated during the **inflation–reheating** transition.  

It also provides quick utilities to check **BBN constraints** and visualize multi-epoch reheating scenarios.

---

## 🚀 Features

- Compute the gravitational-wave spectrum `Ω_GW(f)` for arbitrary reheating histories  
- Handle multiple equation-of-state (EoS) epochs  
- Check Big Bang Nucleosynthesis (BBN) constraints via  
  - `piecewise` (recommended)  
  - `intersection`  
  - `weaker`  
- Minimal dependencies (`numpy`, `scipy`, `matplotlib`)  
- Simple to import — just one Python file!

---

## 📦 Installation

Clone the repository and install locally:

```bash
git clone https://github.com/athul104/GWInSpect.git
cd GWInSpect
pip install -e .
