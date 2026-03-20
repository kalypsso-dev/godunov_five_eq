This directory contains a solver implementation for the so-called `5 equations` model, for compressible two-material flow.

The implementation is a direct adaptation from muscl-hydro, using the same type of finite volume discretization, built upon the HLLC Riemann solver slightly modified for mixed cells.

References:
- [A Five-Equation Model for the Simulation of Interfaces between Compressible Fluids](https://doi.org/10.1006/jcph.2002.7143), Allaire et al., Journal of Computational Physics
Volume 181, Issue 2, 20 September 2002, Pages 577-616.
- [Sharp interface schemes for multi-material computational fluid dynamics](https://doi.org/10.17863/CAM.44907), Cutforth, M. C. (2019), PhD thesis.
