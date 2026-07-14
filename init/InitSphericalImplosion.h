// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitSphericalImplosion.h
 *
 * Spherical implosion test case initialization.
 *
 * Sharp interface schemes for multi-material computational fluid dynamics, PhD thesis
 * Murray Cutforth, University of Cambridge, https://doi.org/10.17863/CAM.44907
 *
 * See test called "tin implosion" presented in chapter 5, section 5.3.8
 *
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_INIT_SPHERICAL_IMPLOSION_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_INIT_SPHERICAL_IMPLOSION_H_

#include <godunov_five_eq/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/SphericalImplosionParams.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve spherical implosion test case.
 *
 * There are 3 regions:
 * - outer region (material 0)
 * - just above the perturbed interface (material 0)
 * - inner region (material 1)
 */
template <size_t dim, typename device_t>
class InitSphericalImplosionDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! this test case has 3 regions (and 2 materials)
  static constexpr int NB_REGIONS = 3;

private:
  //! heavy data
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! SphericalImplosion problem specific parameters (used on device)
  SphericalImplosionParams m_spherical_implosion_params;

  //! Initial states (one per region, conservative variables)
  InitialStates<dim, device_t> m_initial_states;

  //! Number of materials
  int m_num_materials;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! AMR level max
  const int32_t m_level_max;

  InitSphericalImplosionDataFunctor(DataArrayBlock_t const &             Udata,
                                    orchard_key_view_t<device_t> const & orchard_keys,
                                    int32_t                              local_num_octants,
                                    InitialStates<dim, device_t> const & initial_states,
                                    ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_spherical_implosion_params(config_map)
    , m_initial_states(initial_states)
    , m_num_materials(config_map.getInteger("run", "nmat", 0))
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map))
    , m_level_max(config_map.getInteger("amr", "level_max", 0)){};

public:
  //! static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        orchard_key_view_t<device_t> const & orchard_keys,
        int32_t                              local_num_octants,
        InitialStates<dim, device_t> const & m_initial_states,
        ConfigMap const &                    config_map);

  // ====================================================================
  // ====================================================================
  //! initial state in region 1 (inside shell) need to be modified
  //! to add radial velocity.
  KOKKOS_INLINE_FUNCTION
  HydroState<dim>
  update_initial_state1(Kokkos::Array<real_t, dim> const & normal, real_t v_theta) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int32_t & global_index) const;

}; // class InitSphericalImplosionDataFunctor

extern template class InitSphericalImplosionDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitSphericalImplosionDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve SphericalImplosion problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitSphericalImplosionDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitSphericalImplosionRefineFunctor
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

  //! SphericalImplosion problem specific parameters (used on device)
  SphericalImplosionParams m_spherical_implosion_params;

  //! which level should we look at
  int m_level_refine;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

public:
  InitSphericalImplosionRefineFunctor(DataArrayBlock_t const &             Udata,
                                      orchard_key_view_t<device_t> const & orchard_keys,
                                      amrflags_view_t const &              amrflags,
                                      int32_t                              local_num_octants,
                                      int                                  level_refine,
                                      ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_spherical_implosion_params(config_map)
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

}; // class InitSphericalImplosionRefineFunctor

extern template class InitSphericalImplosionRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitSphericalImplosionRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * \class InitSphericalImplosion
 *
 * spherical implosion init.
 */
template <size_t dim, typename device_t>
class InitSphericalImplosion
{
public:
  static void
  apply(SolverGodunovFiveEq<dim, device_t> & solver);
};

extern template class InitSphericalImplosion<2, kalypsso::DefaultDevice>;
extern template class InitSphericalImplosion<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_INIT_SPHERICAL_IMPLOSION_H_
