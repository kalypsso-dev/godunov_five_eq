# Godunov\_five\_eq

This repository is part of [kalypsso-dev](https://github.com/kalypsso-dev) project and is not intended to be standalone, but rather used as a submodule of [kalypsso-app-pub](https://github.com/kalypsso-dev/kalypsso-app-pub).

## What is it ?

This repository contains a solver implementation for the so-called `5 equations` model, for compressible two-material flow.

The implementation is a direct adaptation from godunov\_hydro, using the same type of finite volume discretization, built upon the HLLC Riemann solver slightly modified for mixed cells.

## Scientific references

- [A Five-Equation Model for the Simulation of Interfaces between Compressible Fluids](https://doi.org/10.1006/jcph.2002.7143), Allaire et al., Journal of Computational Physics
Volume 181, Issue 2, 20 September 2002, Pages 577-616.
- [Sharp interface schemes for multi-material computational fluid dynamics](https://doi.org/10.17863/CAM.44907), Cutforth, M. C. (2019), PhD thesis.


# Licenses

This project is released under [Apache-2.0 WITH LLVM-exception](https://github.com/kalypsso-dev/godunov_hydro/LICENSES/Apache-2.0.txt).
