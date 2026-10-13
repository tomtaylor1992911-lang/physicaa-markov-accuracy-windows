# Markov accuracy windows in finite quantum reservoirs

Python code and numerical source data for the manuscript **Tolerance-dependent Markov accuracy windows in finite quantum reservoirs**, by Xiaxia Cui, Renxin Yu, Qiang Tang, and Jau Tang.

This repository reproduces the finite-mode Wigner–Weisskopf comparisons, tolerance-dependent first-exit windows, and residual-based population bounds. It contains four active Python scripts, all 16 numerical CSV/JSON files, and the 10 figure sets in PDF, SVG, and PNG. No experimental data are used. Manuscript files, editorial reviews, and build files are not included.

## Install

Use Python 3.10 or newer. Run these commands from the repository root:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then install the two numerical and plotting dependencies:

```bash
python -m pip install -r requirements.txt
```

## Reproduce

Regenerate all source data and figures:

```bash
python code/generate_figures.py
```

Render from the included data without recalculating the numerical results:

```bash
python code/generate_figures.py --plot-only
```

Calculate source data without rendering figures:

```bash
python code/generate_figures.py --compute-only
```

The complete entry point runs `revision_numerics.py` followed by `polish_analysis.py`. Either script can also be run directly with the same flags; the additional analysis uses baseline data included in this repository. The schematic can be rendered separately with `python code/generate_approximation_map.py`.

Data and numerical summaries are written to `data/`, figures to `figures/`, and duplicate verification summaries to `verification/`. Computation overwrites the corresponding included data; rendering overwrites the corresponding figure exports. Timings in newly computed JSON summaries depend on hardware. See [code/README.md](code/README.md) for the method, file inventory, and numerical scope.

## Numerical scope and provenance

The model uses a finite Lorentzian band and an invariant single-excitation sector, with units setting ℏ = 1. The continuous finite-band comparison is a converged Gauss–Legendre quadrature reference. First-exit searches begin at time zero. A bound above tolerance ends a sufficient interval; it does not demonstrate that the observed error has crossed the tolerance. Numerical quadrature convergence checks do not constitute formal interval-arithmetic certification, and the results are specific to the stated scalar model.

This publication copy was prepared from `PhysicaA_Submission_Package/Polish_20261013/Source`. Numerical calculations and parameters are unchanged. The only executable-code adjustment redirects duplicate summaries from the surrounding submission package to the repository-local `verification/` directory. Documentation was adapted for this standalone repository, and an obsolete provenance reference was removed from the wrapper docstring. Earlier inactive implementations are excluded.
