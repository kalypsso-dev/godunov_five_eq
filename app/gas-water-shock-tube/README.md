# Gas-Water shock tube

## Reference

There are multiple references, we just give one
A hybrid WENO5IS-THINC reconstruction scheme for compressible multiphase flows, Zhang et al., JCP 498 (2024), 112672. https://doi.org/10.1016/j.jcp.2023.112672

## Run

```shell
../solver_godunov_five_eq --ini ./test_gas_water_shock_tube_2d.ini
# or
../solver_godunov_five_eq --ini ./test_gas_water_shock_tube_2d_thinc.ini
```

## Plot

```shell
./plot.py
```
