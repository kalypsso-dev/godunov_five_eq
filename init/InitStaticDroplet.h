// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitStaticDroplet.h
 *
 * Static droplet test case initialization.
 *
 * This test is useful for checking curvature computation and surface tension effect.
 * The initial state is determined so that the Laplace law is verified, i.e. the pressure difference
 * between liquid and gas is balanced by surface tension force. This equilibrium should be
 * maintained by the solver.
 *
 * This test is defined in:
 *
 * A finite-volume HLLC-based scheme for compressible interfacial flows with surface tension,
 * Garrick Owkes and Regele, Journal of Computational Physics Volume 339, 15 June 2017, Pages 46-67.
 * https://doi.org/10.1016/j.jcp.2017.03.007
 *
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_INIT_STATICDROPLET_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_INIT_STATICDROPLET_H_

#include <godunov_five_eq/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/StaticDropletParams.h>

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
class InitStaticDropletDataFunctor
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

  //! StaticDroplet problem specific parameters (used on device)
  StaticDropletParams m_staticDropletParams;

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

public:
  InitStaticDropletDataFunctor(DataArrayBlock_t const &             Udata,
                               orchard_key_view_t<device_t> const & orchard_keys,
                               int32_t                              local_num_octants,
                               InitialStates<dim, device_t> const & initial_states,
                               ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_staticDropletParams(config_map)
    , m_initial_states(initial_states)
    , m_num_materials(config_map.getInteger("run", "nmat", 0))
    , m_eos_wrapper(config_map)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map))
    , m_level_max(config_map.getInteger("amr", "level_max", 0)){};

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

}; // class InitStaticDropletDataFunctor

extern template class InitStaticDropletDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitStaticDropletDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve StaticDroplet problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitStaticDropletDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitStaticDropletRefineFunctor
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

  //! StaticDroplet problem specific parameters (used on device)
  StaticDropletParams m_staticDropletParams;

  //! which level should we look at
  int m_level_refine;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

public:
  InitStaticDropletRefineFunctor(DataArrayBlock_t const &             Udata,
                                 orchard_key_view_t<device_t> const & orchard_keys,
                                 amrflags_view_t const &              amrflags,
                                 int32_t                              local_num_octants,
                                 int                                  level_refine,
                                 ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_staticDropletParams(config_map)
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

}; // class InitStaticDropletRefineFunctor

extern template class InitStaticDropletRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitStaticDropletRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * \class InitStaticDroplet
 *
 * static droplet init.
 */
template <size_t dim, typename device_t>
class InitStaticDroplet
{
public:
  static void
  apply(SolverGodunovFiveEq<dim, device_t> & solver);
};

extern template class InitStaticDroplet<2, kalypsso::DefaultDevice>;
extern template class InitStaticDroplet<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_INIT_STATICDROPLET_H_
