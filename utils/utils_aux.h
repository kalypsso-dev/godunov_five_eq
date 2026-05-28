// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file utils_aux.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_UTILS_AUX_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_UTILS_AUX_H_

#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_HDF5, ...
#include <kalypsso/core/HydroParams.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/DataArrayBlock.h>

#include <godunov_five_eq/models/FiveEq.h>

namespace kalypsso
{

namespace godunov_five_eq
{

/**
 * Compute mixture density.
 *
 * \param[in] Conservative variables array
 *
 * \tparam dim is dimension (2 or 3)
 */
template <size_t dim, typename device_t>
auto
compute_mixture_density(DataArrayBlock<dim, real_t, device_t> U)
  -> DataArrayBlock<dim, real_t, device_t>
{
  using ExecutionSpace = typename device_t::execution_space;

  const auto nbOcts = U.num_quadrants();
  const auto nbCells = U.num_cells();

  auto res = DataArrayBlock<dim, real_t, device_t>("rho_mixture", U.block_size(), 1, nbOcts);

  Kokkos::parallel_for(
    "compute_mixture_density",
    Kokkos::RangePolicy<ExecutionSpace>(0, nbCells * nbOcts),
    KOKKOS_LAMBDA(int32_t global_index) {
      const auto iOct = global_index / nbCells;
      const auto cell_index = global_index - iOct * nbCells;
      res(cell_index, 0, iOct) = U(cell_index, models::FiveEq<dim>::IAD0, iOct) +
                                 U(cell_index, models::FiveEq<dim>::IAD1, iOct);
    });

  return res;

} // compute_mixture_density

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_UTILS_AUX_H_
