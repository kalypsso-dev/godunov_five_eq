// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitTwoFluidShockTube.h
 *
 * Kind of equivalent of Sod shock tube (mono fluid) but adapted to two-fluid flow.
 *
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_INIT_TWOFLUIDSHOCKTUBE_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_INIT_TWOFLUIDSHOCKTUBE_H_

#include <godunov_five_eq/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/TwoFluidShockTubeParams.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// ==================================================================================
// ==================================================================================
// ==================================================================================
/**
 * Implement user data initialization to solve a two fluid shock tube problem.
 *
 * Initial conditions is refined near strong density gradients.
 */
template <size_t dim, typename device_t>
class InitTwoFluidShockTubeDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! a shock tube problem has 2 regions
  static constexpr int NB_REGIONS = 2;

private:
  //! heavy data
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! TwoFluidShockTube problem specific parameters (used on device)
  TwoFluidShockTubeParams m_shock_params;

  //! Initial states (one per region, conservative variables)
  InitialStates<dim, device_t> m_initial_states;

  //! Number of materials
  int m_num_materials;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitTwoFluidShockTubeDataFunctor(DataArrayBlock_t const &             Udata,
                                   orchard_key_view_t<device_t> const & orchard_keys,
                                   int32_t                              local_num_octants,
                                   InitialStates<dim, device_t> const & initial_states,
                                   ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_shock_params(config_map)
    , m_initial_states(initial_states)
    , m_num_materials(config_map.getInteger("run", "nmat", 0))
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

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

}; // InitTwoFluidShockTubeDataFunctor

extern template class InitTwoFluidShockTubeDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitTwoFluidShockTubeDataFunctor<3, kalypsso::DefaultDevice>;

// ==================================================================
// ==================================================================
// ==================================================================
/**
 * Implement initial refinement to solve TwoFluidShockTube problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitTwoFluidShockTubeDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitTwoFluidShockTubeRefineFunctor
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

  //! TwoFluidShockTube problem specific parameters (used on device)
  TwoFluidShockTubeParams m_shock_params;

  //! which level should we look at
  int m_level_refine;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitTwoFluidShockTubeRefineFunctor(DataArrayBlock_t const &             Udata,
                                     orchard_key_view_t<device_t> const & orchard_keys,
                                     amrflags_view_t const &              amrflags,
                                     int32_t                              local_num_octants,
                                     int                                  level_refine,
                                     ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_shock_params(config_map)
    , m_level_refine(level_refine)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

public:
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
  KOKKOS_INLINE_FUNCTION void
  operator()(TagRefineAlways const &, const size_t & iOct) const;

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineGeometric const &, const size_t & iOct) const;

}; // InitTwoFluidShockTubeRefineFunctor

extern template class InitTwoFluidShockTubeRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitTwoFluidShockTubeRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * \class InitTwoFluidShockTube
 *
 * Two fluid sod shock tube init.
 */
template <size_t dim, typename device_t>
class InitTwoFluidShockTube
{
public:
  static void
  apply(SolverGodunovFiveEq<dim, device_t> & solver);
};

extern template class InitTwoFluidShockTube<2, kalypsso::DefaultDevice>;
extern template class InitTwoFluidShockTube<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_INIT_TWOFLUIDSHOCKTUBE_H_
