# GWInSpect (WIP)

**Status:** early alpha, API may change. A minimal, research-oriented Python package to:
- build present-day GW spectra across piecewise-constant reheating epochs, and
- evaluate a piecewise **BBN** constraint integral.

The code supports per-epoch equations of state \(w_i\), energy/temperature transitions, and lets you set the inflation scale via \(r\) **or** \(E_{\rm inf}\). A tabulated \(g_\*(T)\), \(g_{s}(T)\) is bundled.

> 📝 This is work-in-progress accompanying a companion paper on GWs during reheating. Expect rapid changes; validation and tests will follow after the arXiv version is posted.

## Install (source)
```bash
# clone your repo, then in the repo root:
pip install -e .
