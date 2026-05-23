// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeDerivedQuantities.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_COMPUTE_DERIVED_QUANTITIES_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_COMPUTE_DERIVED_QUANTITIES_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock

#include <godunov_five_eq/common.h>
#include <godunov_five_eq/models/FiveEq.h>
#include <godunov_five_eq/models/HydroState.h>
#include <godunov_five_eq/models/utils.h>
#include <godunov_five_eq/eos/eos_utils.h>

namespace kalypsso
{

namespace godunov_five_eq
{

/**
 * Define a better enum to list supported derived quantities.
 *
 * Derived quantities are quantities that are not solved in the PDE systems but can be
 * deduced from conservative variables.
 *
 * Derived quantities can be scalar or vector valued.
 *
 * - (mixture) density
 * - (mixture) thermal pressure
 * - (mixture) speed of sound
 * - (mixture) specific kinetic energy
 * - (mixture) local Mach number : M =|u|/c where c is local (mixture) speed of sound
 */
// clang-format off
BETTER_ENUM(DERIVED_QUANTITY, uint32_t,
            RHO_MIX,
            THERMAL_PRESSURE,
            SPEED_OF_SOUND,
            SPECIFIC_EKIN,
            LOCAL_MACH_NUMBER
  )
// clang-format on

// ========================================================
// ========================================================
// ========================================================
/**
 * A simple helper structure to compute a derived quantity (thermal pressure, specific kinetic
 * energy, ...)
 */
template <size_t dim, typename device_t>
struct ComputeDerivedQuantities
{

  //! type alias for cell-centered data array at block level (see kalypsso_data_container.h)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using ExecutionSpace = typename device_t::execution_space;

  //! make enum FiveEq available
  using FiveEq = models::FiveEq<dim>;

  // ==========================================================================
  // ==========================================================================
  static void
  check_args_validity(DataArrayBlock_t const & Udata,
                      int64_t const &          iOct_begin,
                      int64_t const &          num_octs)
  {

    if (iOct_begin < 0 or iOct_begin >= Udata.num_quadrants())
    {
      Kokkos::abort("[ComputeDerivedQuantities::run] : invalid argument for iOct_begin.");
    }
    if (iOct_begin + num_octs < 0 or (iOct_begin + num_octs) > Udata.num_quadrants())
    {
      Kokkos::abort("[ComputeDerivedQuantities::run] : invalid argument for num_octs.");
    }

  } // check_args_validity

  // ==========================================================================
  // ==========================================================================
  static DataArrayBlock_t
  run(DataArrayBlock_t                     Udata,
      DERIVED_QUANTITY                     quantity,
      HydroSettings                        hydro_settings,
      EosWrapper_t<device_t>               eos,
      int64_t                              iOct_begin,
      int64_t                              num_octs,
      [[maybe_unused]] ParallelEnv const & par_env)
  {
    check_args_validity(Udata, iOct_begin, num_octs);

    const auto label = std::string("compute derived quantity ") + quantity._to_string();

    auto res = DataArrayBlock_t(label, Udata.block_size(), 1, Udata.num_quadrants());

    const auto    nbCellsPerLeaf = Udata.num_cells();
    const int64_t total_num_cells = nbCellsPerLeaf * num_octs;

    Kokkos::parallel_for(
      label,
      Kokkos::RangePolicy<ExecutionSpace>(0, total_num_cells),
      KOKKOS_LAMBDA(const int64_t & global_index) {
        /// convert global index into
        // - octant id
        // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
        const auto iOct = iOct_begin + global_index / nbCellsPerLeaf;
        const auto cell_index =
          static_cast<int32_t>(global_index - (iOct - iOct_begin) * nbCellsPerLeaf);

        // cell-centered conservative variables in current cell
        const auto uLoc = get_conservative_variables(Udata, cell_index, iOct);

        if (quantity._to_integral() == +DERIVED_QUANTITY::RHO_MIX)
        {
          res(cell_index, 0, iOct) = uLoc[FiveEq::ID];
        }
        else if (quantity._to_integral() == +DERIVED_QUANTITY::THERMAL_PRESSURE)
        {
          res(cell_index, 0, iOct) =
            models::compute_mixture_pressure<dim>(uLoc, hydro_settings, eos);
        }
        else if (quantity._to_integral() == +DERIVED_QUANTITY::SPEED_OF_SOUND)
        {
          HydroState<dim> qLoc;
          res(cell_index, 0, iOct) =
            models::computePrimitives<dim>(uLoc, qLoc, hydro_settings, eos);
        }
        else if (quantity._to_integral() == +DERIVED_QUANTITY::SPECIFIC_EKIN)
        {
          res(cell_index, 0, iOct) = models::compute_mixture_ekin<dim>(uLoc, hydro_settings);
        }
        else if (quantity._to_integral() == +DERIVED_QUANTITY::LOCAL_MACH_NUMBER)
        {
          HydroState<dim> qLoc;

          // compute speed of sound
          const auto cs = models::computePrimitives<dim>(uLoc, qLoc, hydro_settings, eos);

          auto u_norm = qLoc[FiveEq::IU] * qLoc[FiveEq::IU] + qLoc[FiveEq::IV] * qLoc[FiveEq::IV];
          if constexpr (dim == 3)
            u_norm += qLoc[FiveEq::IW] * qLoc[FiveEq::IW];
          u_norm = sqrt(u_norm);

          res(cell_index, 0, iOct) = u_norm / cs;
        }
      });

    return res;

  } // run

  // ==========================================================================
  // ==========================================================================
  static DataArrayBlock_t
  run(DataArrayBlock_t       Udata,
      std::string            quantity,
      HydroSettings          hydro_settings,
      EosWrapper_t<device_t> eos,
      int64_t                iOct_begin,
      int64_t                num_octs,
      ParallelEnv const &    par_env)
  {
    if (quantity == "rho_mix")
    {
      return run(
        Udata, DERIVED_QUANTITY::RHO_MIX, hydro_settings, eos, iOct_begin, num_octs, par_env);
    }
    else if (quantity == "thermal_pressure")
    {
      return run(Udata,
                 DERIVED_QUANTITY::THERMAL_PRESSURE,
                 hydro_settings,
                 eos,
                 iOct_begin,
                 num_octs,
                 par_env);
    }
    else if (quantity == "speed_of_sound")
    {
      return run(Udata,
                 DERIVED_QUANTITY::SPEED_OF_SOUND,
                 hydro_settings,
                 eos,
                 iOct_begin,
                 num_octs,
                 par_env);
    }
    else if (quantity == "specific_ekin")
    {
      return run(
        Udata, DERIVED_QUANTITY::SPECIFIC_EKIN, hydro_settings, eos, iOct_begin, num_octs, par_env);
    }
    else if (quantity == "local_mach_number")
    {
      return run(Udata,
                 DERIVED_QUANTITY::LOCAL_MACH_NUMBER,
                 hydro_settings,
                 eos,
                 iOct_begin,
                 num_octs,
                 par_env);
    }
    else
    {
      KALYPSSO_ERROR(
        "ComputeDerivedQuantity: unknow quantity (check your input parameter file) - use "
        "thermal pressure instead.");
      return run(Udata,
                 DERIVED_QUANTITY::THERMAL_PRESSURE,
                 hydro_settings,
                 eos,
                 iOct_begin,
                 num_octs,
                 par_env);
    }
  } // run

}; // struct ComputeDerivedQuantities

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_COMPUTE_DERIVED_QUANTITIES_H_
