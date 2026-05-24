// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitShockBubble.h
 *
 * Shock-bubble interaction.
 *
 * Reference: http://amroc.sourceforge.net/examples/euler/2d/html/shbubble_n.htm
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_INIT_SHOCKBUBBLE_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_INIT_SHOCKBUBBLE_H_

#include <godunov_five_eq/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/ShockBubbleParams.h>
#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve a shock-bubble interaction problem.
 *
 * \sa references
 * - http://amroc.sourceforge.net/examples/euler/2d/html/shbubble_n.htm
 *
 * Initial conditions is refined near strong density gradients.
 */
template <size_t dim, typename device_t>
class InitShockBubbleDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

private:
  //! heavy data
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! ShockBubble problem specific parameters (used on device)
  ShockBubbleParams<device_t> m_sb_params;

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

  InitShockBubbleDataFunctor(DataArrayBlock_t const &             Udata,
                             orchard_key_view_t<device_t> const & orchard_keys,
                             int32_t                              local_num_octants,
                             InitialStates<dim, device_t> const & initial_states,
                             ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_sb_params(config_map)
    , m_initial_states(initial_states)
    , m_num_materials(config_map.getInteger("run", "nmat", 0))
    , m_eos_wrapper(config_map)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

public:
  //! static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        orchard_key_view_t<device_t> const & orchard_keys,
        int32_t                              local_num_octants,
        InitialStates<dim, device_t> const & initial_states,
        ConfigMap const &                    config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int32_t & global_index) const;

private:
  // ==========================================================================
  // ==========================================================================
  /**
   * Utility to determine in which region a given point is.
   */
  KOKKOS_INLINE_FUNCTION int
  point_to_region(Kokkos::Array<real_t, dim> const & xyz) const
  {
    auto const & bubble_radius = m_sb_params.bubble_radius;
    auto const & bubble_x = m_sb_params.bubble_x;
    auto const & bubble_y = m_sb_params.bubble_y;
    auto const & bubble_z = m_sb_params.bubble_z;

    // see if point is inside of the bubbles
    for (int i = 0; i < m_sb_params.num_bubbles; ++i)
    {
      auto r2 = (bubble_x(i) - xyz[IX]) * (bubble_x(i) - xyz[IX]) +
                (bubble_y(i) - xyz[IY]) * (bubble_y(i) - xyz[IY]);
      if constexpr (dim == 3)
        r2 += (bubble_z(i) - xyz[IZ]) * (bubble_z(i) - xyz[IZ]);

      if (r2 < bubble_radius(i) * bubble_radius(i))
        return i + 2;
    }

    // is point in post-shock region ?
    if (xyz[IX] < m_sb_params.x_front)
      return 0;

    // point can only be in pre-shock region
    return 1;
  } // point_to_region

  // ==========================================================================
  // ==========================================================================
  /**
   * For each corner of a given cell, determine in which region this corner is.
   *
   * \param[in] ijk i-j-k indexes of a cell inside a block of cells
   * \param[in] key orchard key of current octant
   * \param[out] regions array of regions (one per corner)
   */
  KOKKOS_INLINE_FUNCTION void
  compute_corner_to_region(coord_t<dim> const &                             ijk,
                           key_t const &                                    key,
                           block_size_t<dim> const &                        block_sizes,
                           Kokkos::Array<int, Corner::num_corners<dim>()> & regions) const
  {

    for (uint8_t i_corner = 0; i_corner < Corner::num_corners<dim>(); i_corner++)
    {
      const auto xyz_vertex_corner =
        orchard_key_to_corner_coord<dim>(key, ijk, block_sizes[IX], i_corner);
      auto xyz_corner =
        vertex_coord_to_real_space<dim>(xyz_vertex_corner, m_scaling_factor, m_xyz_min);

      regions[i_corner] = point_to_region(xyz_corner);
    }
  }

}; // class InitShockBubbleDataFunctor

extern template class InitShockBubbleDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitShockBubbleDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve ShockBubble problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitShockBubbleDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitShockBubbleRefineFunctor
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

  //! ShockBubble problem specific parameters (used on device)
  ShockBubbleParams<device_t> m_sb_params;

  //! which level should we look at
  int m_level_refine;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitShockBubbleRefineFunctor(DataArrayBlock_t const &             Udata,
                               orchard_key_view_t<device_t> const & orchard_keys,
                               amrflags_view_t const &              amrflags,
                               int32_t                              local_num_octants,
                               int                                  level_refine,
                               ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_sb_params(config_map)
    , m_level_refine(level_refine)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

public:
  //! static method which does it all: create and execute functor
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

}; // class InitShockBubbleRefineFunctor

extern template class InitShockBubbleRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitShockBubbleRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * \class InitShockBubble
 *
 * Shock-bubble init.
 */
template <size_t dim, typename device_t>
class InitShockBubble
{
public:
  static void
  apply(SolverGodunovFiveEq<dim, device_t> & solver);

}; // class InitShockBubble

extern template class InitShockBubble<2, kalypsso::DefaultDevice>;
extern template class InitShockBubble<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_INIT_SHOCKBUBBLE_H_
