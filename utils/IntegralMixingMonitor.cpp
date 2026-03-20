// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file IntegralMixingMonitor.cpp
 *
 * \brief Contains the definition of IntegralMixingMonitor.
 */

#include <godunov_five_eq/utils/IntegralMixingMonitor.h>

#include <kalypsso/core/cnpy_io.h>
#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
IntegralMixingMonitor<dim, device_t>::record(const DataArrayBlock<dim, real_t, device_t> & data,
                                             const OrchardKeys &                           keys,
                                             const int32_t                        start_octant,
                                             const int32_t                        end_octant,
                                             int32_t                              i_phi,
                                             real_t                               current_time,
                                             [[maybe_unused]] const ParallelEnv & par_env)
{

  const int32_t                       nb_cells = data.num_cells();
  const int32_t                       start = start_octant * nb_cells;
  const int32_t                       end = end_octant * nb_cells;
  Kokkos::RangePolicy<ExecutionSpace> policy(start, end);

  const auto scaling_factor = m_scaling_factor;

  real_t              local_total = 0;
  Kokkos::Sum<real_t> reducer(local_total);
  Kokkos::parallel_reduce(
    "kalypsso::godunov_five_eq::IntegralMixingMonitor",
    policy,
    KOKKOS_LAMBDA(const int32_t & i_global, real_t & total) {
      const auto block_size = data.block_size();
      // const auto nb_cells = data.num_cells();
      const auto i_oct = i_global / nb_cells;
      const auto i_cell = i_global - nb_cells * i_oct;

      const auto level = orchard_key_t<dim>::level(keys(i_oct));
      const auto dx = compute_cell_length<dim>(level, block_size[IX]) * scaling_factor;

      real_t vol = dx * dx;
      if constexpr (dim == 3)
        vol *= dx;

      total += vol * data(i_cell, i_phi, i_oct) * (ONE_F - data(i_cell, i_phi, i_oct));
    },
    reducer);

  real_t global_total = local_total;
#ifdef KALYPSSO_CORE_USE_MPI
  par_env.comm().template MPI_Allreduce<MpiComm::SUM>(&local_total, &global_total, 1);
#endif // KALYPSSO_CORE_USE_MPI

  int32_t & i = m_mixing_params.recorded();
  m_mixings(i) = global_total;
  m_times(i) = current_time;
  i++;

} // IntegralMixingMonitor<dim, device_t>::record

// ================================================================================================
// ================================================================================================
template <size_t dim, typename device_t>
void
IntegralMixingMonitor<dim, device_t>::save()
{
  save_cnpy(m_mixings, m_mixing_params.file_prefix() + "_mixing_integrals");
  save_cnpy(m_times, m_mixing_params.file_prefix() + "_mixing_times");
}

// ================================================================================================
// ================================================================================================
template class IntegralMixingMonitor<2, kalypsso::DefaultDevice>;
template class IntegralMixingMonitor<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso
