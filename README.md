# On the Parisi PDE and the Overlap Gap Property in Ising Spin Glasses

## Course Project: 6.7720[J]/18.619[J]/15.070[J] Discrete Probability and Stochastic Processes

[![Preview](https://img.shields.io/badge/Download%20as%20PDF-EF3939?style=flat&logo=adobeacrobatreader&logoColor=white&color=black&labelColor=ec1c24)](https://raw.githubusercontent.com/xenakistheo/SpinGlasses/main/ParisiPDE_OGP_SpinGlass.pdf)

Utkarsh [utkarsh5@mit.edu](mailto:utkarsh5@mit.edu), Theodoros Xenakis [txenaxis@mit.edu](mailto:txenaxis@mit.edu)

We study the overlap gap property (OGP) for the pure p-spin Ising spin glass
through the lens of the zero-temperature Parisi variational principle. The ground-
state energy is characterized by minimizing the Parisi functional over a class of
nondecreasing functions γ, whose flat regions correspond to forbidden intervals in
the limiting overlap distribution—the PDE-level signature of OGP. We prove that
for every pure p > 2 model, no minimizer of the Parisi functional can be strictly
increasing, establishing a PDE-level gap. The argument builds on the work of El
Alaoui, Montanari, and Sellke [EAMS21] and reduces to a scaling incompatibility
between a stationarity identity and the regularity of the Parisi PDE near t = 0.
We also propose a conjectural upgrade to a finite-N binary OGP via constrained
two-replica energies. On the numerical side, we introduce a practical simulation
framework based on the Hopf–Cole formula and Gauss–Hermite quadrature, which
avoids the exponential cost of nested Monte Carlo and reduces PDE solves to
O(KNxQ) operations. Experiments for p = 2, . . . , 7 confirm one-step replica
symmetry breaking (1RSB) for p ≥ 3 and full replica symmetry breaking (FRSB)
for p = 2. Analysis of the Hessian loss landscape reveals that the condition number
grows from ≈ 7.6 at p = 2 to ∼ 1013 at p = 7, providing a sharp numerical
signature of the 1RSB structure.

