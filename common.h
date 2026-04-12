// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file common.h
 *
 * \brief Contains a shared set of values and types.
 */

#ifndef KALYPSSO_GODUNOV_FIVE_EQ_COMMON_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_COMMON_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/DataArrayBlock.h>
#include <godunov_five_eq/models/FiveEq.h>
#include <godunov_five_eq/models/HydroState.h>
#include <godunov_five_eq/eos/eos_utils.h>

#include <kalypsso/core/models/HydroSettings.h>
#include <kalypsso/core/multimaterial_utils.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/AMRContext.h>
#include <kalypsso/core/brick_utils.h>

#include <../better-enums/enum.h>

namespace kalypsso
{

namespace godunov_five_eq
{

template <size_t dim, typename device_t>
class SolverGodunovFiveEq;

//! Array of Orchard keys
template <typename device_t>
using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

//! Shorthand for the five eq model
using FiveEq = models::FiveEq;

//! Hashmap type from Orchard keys to octant index
template <typename device_t>
using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;

//! AMR flags array type
template <size_t dim, typename device_t>
using amrflags_view_t = typename AMRContext<dim, device_t>::amrflags_view_t;

//! Region initial states array
template <size_t dim, typename device_t>
using InitialStates = Kokkos::View<HydroState<dim> *, device_t>;

/**
 * Read initial hydrodynamics state.
 *
 * This is useful when doing an initialization that is constant (i.e. space uniform)
 * inside spatial regions.
 *
 * Currently region shapes are defined by each test case.
 * Soon we will also be able to define region shapes through geometric modelling.
 *
 * \return a vector of conservative variables
 *
 * \param[in] i_region region id
 * \param[in] config_map application's configuration parameters
 */
template <size_t dim>
HydroState<dim>
get_region_init_state(const int32_t                    i_region,
                      EosWrapper_t<HostDevice> const & eos_wrapper,
                      const ConfigMap &                config_map)
{
  // makes enum Hydro::VarId available
  using Hydro = models::FiveEq;

  // variables specific (per mass unit)
  HydroState<dim> q;

  const auto section = "region" + std::to_string(i_region);
  const auto phi_rho = config_map.getRealVector(
    section, "phi_rho", std::vector<real_t>{ KALYPSSO_NUM(0.0), KALYPSSO_NUM(0.0) });
  if (phi_rho.size() != 2)
  {
    Kokkos::abort(
      "Error reading input file: wrong number of values for \"phi_rho\" (must be two !)");
  }
  const auto phi0 = config_map.getReal(section, "phi0", KALYPSSO_NUM(1.0));

  if (phi0 <= ZERO_F and phi_rho[0] > 0)
  {
    Kokkos::abort("Error reading input file: wrong values for \"phi_rho[0]\" (must be zero !)");
  }
  if (phi0 >= ONE_F and phi_rho[1] > 0)
  {
    Kokkos::abort("Error reading input file: wrong values for \"phi_rho[1]\" (must be zero !)");
  }

  q[Hydro::IPHI] = phi0;
  q[Hydro::ID0] = phi_rho[0];
  q[Hydro::ID1] = phi_rho[1];
  q[Hydro::ID] = q[Hydro::ID0] + q[Hydro::ID1];
  q[Hydro::IU] = config_map.getReal(section, "u", KALYPSSO_NUM(0.0));
  q[Hydro::IV] = config_map.getReal(section, "v", KALYPSSO_NUM(0.0));
  if constexpr (dim == 3)
    q[Hydro::IW] = config_map.getReal(section, "w", KALYPSSO_NUM(0.0));

  const auto p = config_map.getReal(section, "p", KALYPSSO_NUM(1.0));
  assertm(phi_rho[0] >= 0, "Invalid value for partial density 0");
  assertm(phi_rho[1] >= 0, "Invalid value for partial density 1");
  assertm(p >= 0, "Invalid value for pressure");

  const auto & rho = q[Hydro::ID];

  // mixed state
  const auto eint_specific =
    eos_wrapper.mixture_specific_eint(rho, p, phi0, 1 - phi0, phi_rho[0], phi_rho[1]);

  const real_t ekin_specific = [](HydroState<dim> & qq) {
    if constexpr (dim == 2)
      return HALF_F * (qq[Hydro::IU] * qq[Hydro::IU] + qq[Hydro::IV] * qq[Hydro::IV]);
    else if constexpr (dim == 3)
      return HALF_F * (qq[Hydro::IU] * qq[Hydro::IU] + qq[Hydro::IV] * qq[Hydro::IV] +
                       qq[Hydro::IW] * qq[Hydro::IW]);
  }(q);

  // compute conservative variables
  HydroState<dim> Ucons;

  Ucons[Hydro::ID] = rho;
  Ucons[Hydro::IPHI] = phi0;
  Ucons[Hydro::ID0] = q[Hydro::ID0];
  Ucons[Hydro::ID1] = q[Hydro::ID1];
  Ucons[Hydro::IE] = (eint_specific + ekin_specific) * rho;
  Ucons[Hydro::IU] = q[Hydro::IU] * rho;
  Ucons[Hydro::IV] = q[Hydro::IV] * rho;
  if constexpr (dim == 3)
    Ucons[Hydro::IW] = q[Hydro::IW] * rho;

  return Ucons;

} // get_region_init_state

/**
 * Get an array of initiaal states (one per region).
 *
 * This is useful for all simple test case that are initialized with a uniform state per region.
 */
template <size_t dim, typename device_t>
auto
get_initial_states(ConfigMap const & config_map, int nb_regions) -> InitialStates<dim, device_t>
{
  using Hydro = models::FiveEq;
  InitialStates<dim, device_t> initial_states("Initial states", static_cast<uint>(nb_regions));
  auto                         initial_states_host = Kokkos::create_mirror_view(initial_states);

  const auto eos_wrapper = EosWrapper_t<HostDevice>(config_map);

  for (int i_region = 0; i_region < nb_regions; i_region++)
  {
    initial_states_host(i_region) = get_region_init_state<dim>(i_region, eos_wrapper, config_map);
  }
  Kokkos::deep_copy(initial_states, initial_states_host);

  return initial_states;
}

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_COMMON_H_
