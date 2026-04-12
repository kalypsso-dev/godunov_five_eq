// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitUnderwaterExplosion.h
 *
 * Underwater explosion interaction.
 *
 * References:
 * - An interface capturing scheme for modeling atomization in compressible flows,
 * Daniel P. Garrick et al, Journal of Computational Physics Volume 344, 1 September 2017,
 * Pages 260-280. https://doi.org/10.1016/j.jcp.2017.04.079
 *
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_INIT_UNDERWATEREXPLOSION_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_INIT_UNDERWATEREXPLOSION_H_

#include <godunov_five_eq/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/UnderwaterExplosionParams.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve underwater explosion problem.
 *
 * Initial conditions is refined near strong density gradients.
 */
template <size_t dim, typename device_t>
class InitUnderwaterExplosionDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! this test case has 3 regions (air above, water below, air inside bubble)
  static constexpr int NB_REGIONS = 3;

private:
  //! heavy data
  DataArrayBlock_t m_Udata;

  //! field manager
  FieldMap<models::FiveEq> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! UnderwaterExplosion problem specific parameters (used on device)
  UnderwaterExplosionParams m_underwater_explosion_params;

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

public:
  InitUnderwaterExplosionDataFunctor(DataArrayBlock_t                     Udata,
                                     FieldMap<models::FiveEq>             fm,
                                     orchard_key_view_t<device_t>         orchard_keys,
                                     int32_t                              local_num_octants,
                                     InitialStates<dim, device_t> const & initial_states,
                                     ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_fm(fm)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_underwater_explosion_params(config_map)
    , m_initial_states(initial_states)
    , m_num_materials(config_map.getInteger("run", "nmat", 0))
    , m_eos_wrapper(config_map)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t                     Udata,
        FieldMap<models::FiveEq>             fm,
        orchard_key_view_t<device_t>         orchard_keys,
        int32_t                              local_num_octants,
        InitialStates<dim, device_t> const & m_initial_states,
        ConfigMap const &                    config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int32_t & global_index) const;

}; // class InitUnderwaterExplosionDataFunctor

extern template class InitUnderwaterExplosionDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitUnderwaterExplosionDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve UnderwaterExplosion problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitUnderwaterExplosionDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitUnderwaterExplosionRefineFunctor
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

  //! field manager
  FieldMap<models::FiveEq> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! UnderwaterExplosion problem specific parameters (used on device)
  UnderwaterExplosionParams m_underwater_explosion_params;

  //! which level should we look at
  int m_level_refine;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

public:
  InitUnderwaterExplosionRefineFunctor(DataArrayBlock_t             Udata,
                                       FieldMap<models::FiveEq>     fm,
                                       orchard_key_view_t<device_t> orchard_keys,
                                       amrflags_view_t              amrflags,
                                       int32_t                      local_num_octants,
                                       int                          level_refine,
                                       ConfigMap const &            config_map)
    : m_Udata(Udata)
    , m_fm(fm)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_underwater_explosion_params(config_map)
    , m_level_refine(level_refine)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t             Udata,
        FieldMap<models::FiveEq>     fm,
        orchard_key_view_t<device_t> orchard_keys,
        amrflags_view_t              amrflags,
        int32_t                      local_num_octants,
        int                          level_refine,
        ConfigMap const &            config_map);

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

}; // class InitUnderwaterExplosionRefineFunctor

extern template class InitUnderwaterExplosionRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitUnderwaterExplosionRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * \class InitUnderwaterExplosion
 *
 * Underwater explosion init.
 */
template <size_t dim, typename device_t>
class InitUnderwaterExplosion
{
public:
  static void
  apply(SolverGodunovFiveEq<dim, device_t> & solver);
};

extern template class InitUnderwaterExplosion<2, kalypsso::DefaultDevice>;
extern template class InitUnderwaterExplosion<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_INIT_UNDERWATEREXPLOSION_H_
