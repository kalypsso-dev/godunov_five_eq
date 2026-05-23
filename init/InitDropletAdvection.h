// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitDropletAdvection.h
 *
 * Droplet advection test case initialization.
 *
 * This test is useful for testing THINC reconstruction and its ability to maintain a sharp
 * interface over time.
 *
 * This test is discussed in:
 *
 * A hybrid WENO5IS-THINC reconstruction scheme for compressible multiphase flows, Zhand et al.,
 * Journal of Computational Physics Volume 498, 1 February 2024, 112672.
 * https://doi.org/10.1016/j.jcp.2023.112672
 *
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_INIT_DROPLETADVECTION_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_INIT_DROPLETADVECTION_H_

#include <godunov_five_eq/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/DropletAdvectionParams.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve static droplet test case.
 *
 * Initial conditions is refined near strong density gradients.
 */
template <size_t dim, typename device_t>
class InitDropletAdvectionDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! this test case has 2 regions (inside/outside droplet)
  static constexpr int NB_REGIONS = 2;

private:
  //! heavy data
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! DropletAdvection problem specific parameters (used on device)
  DropletAdvectionParams m_droplet_advection_params;

  //! Initial states (one per region, conservative variables)
  InitialStates<dim, device_t> m_initial_states;

  //! Number of materials
  int m_num_materials;

  //! Equation of state wrapper
  EosWrapper_t<device_t> m_eos_wrapper;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! AMR level max
  const int32_t m_level_max;

  InitDropletAdvectionDataFunctor(DataArrayBlock_t const &             Udata,
                                  orchard_key_view_t<device_t> const & orchard_keys,
                                  int32_t                              local_num_octants,
                                  InitialStates<dim, device_t> const & initial_states,
                                  ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_droplet_advection_params(config_map)
    , m_initial_states(initial_states)
    , m_num_materials(config_map.getInteger("run", "nmat", 0))
    , m_eos_wrapper(config_map)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map))
    , m_level_max(config_map.getInteger("amr", "level_max", 0)){};

public:
  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        orchard_key_view_t<device_t> const & orchard_keys,
        int32_t                              local_num_octants,
        InitialStates<dim, device_t> const & m_initial_states,
        ConfigMap const &                    config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int32_t & global_index) const;

}; // class InitDropletAdvectionDataFunctor

extern template class InitDropletAdvectionDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitDropletAdvectionDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve DropletAdvection problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitDropletAdvectionDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitDropletAdvectionRefineFunctor
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! type alias for a (device) Kokkos view of refinement flags
  using amrflags_view_t = typename AMRContext<dim, device_t>::amrflags_view_t;

  struct TagRefineAlways
  {};
  struct TagRefineGeometric
  {};

private:
  //! heavy hydrodynamics data
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! DropletAdvection problem specific parameters (used on device)
  DropletAdvectionParams m_droplet_advection_params;

  //! which level should we look at
  int m_level_refine;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

public:
  InitDropletAdvectionRefineFunctor(DataArrayBlock_t const &             Udata,
                                    orchard_key_view_t<device_t> const & orchard_keys,
                                    amrflags_view_t const &              amrflags,
                                    int32_t                              local_num_octants,
                                    int                                  level_refine,
                                    ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_droplet_advection_params(config_map)
    , m_level_refine(level_refine)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        orchard_key_view_t<device_t> const & orchard_keys,
        amrflags_view_t const &              amrflags,
        int32_t                              local_num_octants,
        int                                  level_refine,
        ConfigMap const &                    config_map);

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineAlways const &, const size_t & iOct) const;

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineGeometric const &, const size_t & iOct) const;

}; // class InitDropletAdvectionRefineFunctor

extern template class InitDropletAdvectionRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitDropletAdvectionRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * \class InitDropletAdvection
 *
 * static droplet init.
 */
template <size_t dim, typename device_t>
class InitDropletAdvection
{
public:
  static void
  apply(SolverGodunovFiveEq<dim, device_t> & solver);
};

extern template class InitDropletAdvection<2, kalypsso::DefaultDevice>;
extern template class InitDropletAdvection<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_INIT_DROPLETADVECTION_H_
