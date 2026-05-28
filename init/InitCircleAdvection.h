// SPDX-FileCopyrightText: 2026 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitCircleAdvection.h
 *
 * Bi-material circle advection test case initialization.
 *
 * This test is useful for testing THINC reconstruction and its ability to maintain a sharp
 * interface over time.
 *
 * This test is first defined in
 *
 * Reconstructing Volume Tracking, William J. Rider, Douglas B. Kothe,
 * Journal of Computational Physics, Volume 141, Issue 2, 1998, Pages 112-152, ISSN 0021-9991,
 * https://doi.org/10.1006/jcph.1998.5906.
 *
 * and further discussed in, e.g. :
 *
 * Extension of the hybrid WENO5IS-THINC scheme to compressible multiphase flows with an
 * arbitrary number of components, Wenbin Zhang, Thomas Paula, Alexander Bußmann, Stefan Adami,
 * Nikolaus A. Adams, Journal of Computational Physics, Volume 524, 2025, 113702,
 * ISSN 0021-9991, https://doi.org/10.1016/j.jcp.2024.113702.
 * see section 4.1.2 and 4.2
 *
 * Multi-material hydrodynamics with algebraic sharp interface capturing, Aditya K. Pandare,
 * Jacob Waltz, Jozsef Bakosi, Computers & Fluids, Volume 215, 2021, 104804, ISSN 0045-7930,
 * https://doi.org/10.1016/j.compfluid.2020.104804.
 * see section 5.1
 *
 * Extension of generic two-component VOF interface advection schemes to an arbitrary number of
 * components, Matthieu Ancellin, Bruno Després, Stéphane Jaouen,
 * Journal of Computational Physics, Volume 473, 2023, 111721, ISSN 0021-9991,
 * https://doi.org/10.1016/j.jcp.2022.111721.
 * see section 3.5
 *
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_INIT_CIRCLEADVECTION_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_INIT_CIRCLEADVECTION_H_

#include <godunov_five_eq/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/CircleAdvectionParams.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve static circle test case.
 *
 * Initial conditions is refined near strong density gradients.
 */
template <size_t dim, typename device_t>
class InitCircleAdvectionDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! this test case has 2 regions (inside/outside circle)
  static constexpr int NB_REGIONS = 2;

private:
  //! heavy data
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! CircleAdvection problem specific parameters (used on device)
  CircleAdvectionParams m_circle_advection_params;

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

  InitCircleAdvectionDataFunctor(DataArrayBlock_t const &             Udata,
                                 orchard_key_view_t<device_t> const & orchard_keys,
                                 int32_t                              local_num_octants,
                                 InitialStates<dim, device_t> const & initial_states,
                                 ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_circle_advection_params(config_map)
    , m_initial_states(initial_states)
    , m_num_materials(config_map.getInteger("run", "nmat", 0))
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

}; // class InitCircleAdvectionDataFunctor

extern template class InitCircleAdvectionDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitCircleAdvectionDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve CircleAdvection problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitCircleAdvectionDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitCircleAdvectionRefineFunctor
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

  //! CircleAdvection problem specific parameters (used on device)
  CircleAdvectionParams m_circle_advection_params;

  //! which level should we look at
  int m_level_refine;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

public:
  InitCircleAdvectionRefineFunctor(DataArrayBlock_t const &             Udata,
                                   orchard_key_view_t<device_t> const & orchard_keys,
                                   amrflags_view_t const &              amrflags,
                                   int32_t                              local_num_octants,
                                   int                                  level_refine,
                                   ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_circle_advection_params(config_map)
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

}; // class InitCircleAdvectionRefineFunctor

extern template class InitCircleAdvectionRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitCircleAdvectionRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * \class InitCircleAdvection
 *
 * static circle init.
 */
template <size_t dim, typename device_t>
class InitCircleAdvection
{
public:
  static void
  apply(SolverGodunovFiveEq<dim, device_t> & solver);
};

extern template class InitCircleAdvection<2, kalypsso::DefaultDevice>;
extern template class InitCircleAdvection<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_INIT_CIRCLEADVECTION_H_
