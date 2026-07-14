// SPDX-FileCopyrightText: 2026 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file BCJet.h
 *
 * Define a border condition functor that can be used to simulate a high-speed inlet flow inside
 * the computational domain.
 *
 * This functor is aimed to be called after FillOutside, to override values at some location.
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_BC_JET_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_BC_JET_H_

#include <kalypsso/core/FillOutside_utils.h>

#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>

#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>

#include <kalypsso/core/utils_block.h> // for definition of function cellindex_to_coord and coord_to_cellindex
#include <kalypsso/core/AMRMeshInfo.h>

#include <godunov_five_eq/common.h>
#include <godunov_five_eq/models/utils.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// =============================================================================
// =============================================================================
//! \struct BCJet
//!
//! Define a circular inlet on the left x face.
template <size_t dim>
struct BCJet
{

  struct Inflow
  {
    //! position of the center of the jet center along y axis
    real_t y = KALYPSSO_NUM(0.5);

    //! position of the center of the jet center along z axis
    real_t z = KALYPSSO_NUM(0.5);

    //! jet width
    real_t width = KALYPSSO_NUM(0.05);

    //! left border location
    real_t xmin = KALYPSSO_NUM(0.0);
  };

  Inflow inflow;

  BCJet(ConfigMap const & config_map)
  {
    inflow.y = config_map.getReal("jet", "y", KALYPSSO_NUM(0.5));
    inflow.z = config_map.getReal("jet", "z", KALYPSSO_NUM(0.5));
    inflow.width = config_map.getReal("jet", "width", KALYPSSO_NUM(0.05));
    inflow.xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0));
  }

  KOKKOS_FUNCTION bool
  needs_override(real_t x, real_t y) const
  {
    // clang-format off
    return (x < inflow.xmin and
            y > inflow.y - inflow.width / 2 and
            y < inflow.y + inflow.width / 2);
    // clang-format on
  }

  KOKKOS_FUNCTION bool
  needs_override(real_t x, real_t y, real_t z) const
  {
    const auto radius2 = (y - inflow.y) * (y - inflow.y) + (z - inflow.z) * (z - inflow.z);
    const auto width2 = inflow.width * inflow.width;

    return (x < inflow.xmin) and (radius2 < width2 / 4);
  }

  //! return conservative variables - 2d
  KOKKOS_FUNCTION real_t
  operator()([[maybe_unused]] real_t x, [[maybe_unused]] real_t y, int var) const
  {
    return inflow.u[var];
  } // operator() - 2d

  //! return conservative variables - 3d
  KOKKOS_FUNCTION real_t
  operator()([[maybe_unused]] real_t x,
             [[maybe_unused]] real_t y,
             [[maybe_unused]] real_t z,
             int                     var) const
  {
    return inflow.u[var];
  } // operator() - 3d

}; // struct BCJet

// ==============================================================================
// ==============================================================================
/**
 * \class FillOutsideJet
 *
 * Override values in some outside cell to have in inlet jet.
 *
 * In the drawing below, the arrow locates the inlet jet.
 *
 * --------------------------------
 * |              3               |
 * |______________________________|
 * |    |                    |    |
 * |    |                    |    |
 * | -> |                    | 2  |
 * | -> |                    |    |
 * |    |                    |    |
 * |____|____________________|____|
 * |              4               |
 * |______________________________|
 *
 */
template <size_t dim, typename device_t>
class FillOutsideJet
{
public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  // ==============================================================
  // ==============================================================
  /** Fill outside octant cells.
   *
   * \param[in,out] userdata_out data array which we want to fill the block ghosts cells
   * \param[in] amr_info gives the number of owned, ghost, outside, ghost_outside quads
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   * \param[in] config_map application parameter map
   */
  static void
  apply(DataArrayBlock_t const &         userdata,
        AMRMeshInfo const &              amr_mesh_info,
        orchard_key_view_t const &       orchard_keys,
        amr_hashmap_t const &            amr_hashmap,
        EosWrapper_t<HostDevice> const & eos_host,
        ConfigMap const &                config_map);

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor to sweep outside leaves.
   *
   * \see FillOutsideCellFunctor
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(const index_t & i_global) const;

private:
  /**
   *
   * \param[in,out] userdata_out data array which we want to fill the block ghosts cells
   * \param[in] amr_info gives the number of owned, ghost, outside, ghost_outside quads
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   * \param[in] config_map application parameter map
   *
   */
  FillOutsideJet(DataArrayBlock_t const &   userdata,
                 AMRMeshInfo const &        amr_mesh_info,
                 orchard_key_view_t const & orchard_keys,
                 amr_hashmap_t const &      amr_hashmap,
                 ConfigMap const &          config_map,
                 HydroState<dim>            u_jet);

  //! a block data array (no ghosts, sizes= bx,by,bz)
  DataArrayBlock_t m_userdata;

  //! AMR mesh information (number of owned quadrants, number of ghosts quadrants, number of
  //! outside quads, number of outside ghosts, etc...)
  AMRMeshInfo m_amr_mesh_info;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_size;

  //! for each direction, state if mesh is periodic
  const Kokkos::Array<bool, dim> m_brick_periodicity;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! Jet parameters
  BCJet<dim> m_bc_jet;

  //! jet conservative variables
  HydroState<dim> m_u_jet;

}; // class FillOutsideJet

// explicit template instantiation
extern template class FillOutsideJet<2, kalypsso::DefaultDevice>;
extern template class FillOutsideJet<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_BC_JET_H_
