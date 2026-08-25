// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ReadFluxesAndConservativeUpdateFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_READ_FLUXES_AND_CONSERVATIVE_UPDATE_FUNCTOR_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_READ_FLUXES_AND_CONSERVATIVE_UPDATE_FUNCTOR_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/ConformalFaceStatus.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/volume_fraction_advection_source_term_type.h>

// model
#include <godunov_five_eq/common.h>
#include <godunov_five_eq/models/utils.h>
#include <godunov_five_eq/models/HydroState.h>
#include <godunov_five_eq/eos/eos_utils.h>

#include <type_traits>

namespace kalypsso
{

namespace godunov_five_eq
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Read fluxes (on conservative variables) and perform a CONSERVATIVE update of conservative
 * variables (i.e. perform time integration using MUSCL Godunov scheme).
 *
 * Input data is Ugroup (containing ghosted block data)
 *
 * We compute fluxes (using Riemann solver) and perform
 * update directly in external array U.
 * Loop through all cell (sub-)faces.
 *
 * \note This functor actually assumes the slopes array to be ghosted array with ghost width of 1.
 * Conservative variable array is assume to be a block array (no ghost).
 *
 * \todo routines like reconstruct_state_2d/3d could probably be
 * moved outside to alleviate this class.
 *
 */
template <size_t dim, typename device_t>
class ReadFluxesAndConservativeUpdateFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  // hashmap related type aliases
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using conformal_status_view_type = conformal_status_view_t<dim, device_t>;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using Hydro = models::FiveEq<dim>;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

  using CellLocation_t = CellLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

private:
  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! conformal status view
  conformal_status_view_type m_conformal_status;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! user data - entire mesh - input (at time \f$t_n\f$)
  DataArrayBlock_t m_Uin;

  //! user data - entire mesh - in/out (at time \f$t_{n+1}\f$)
  DataArrayBlock_t m_Uout;

  //! fluxes - owned and ghost quadrants
  DataArrayBlock_t m_Fluxes;

  //! star velocity (output)
  DataArrayBlock_t m_u_star;

  //! flux direction (IX, IY or IZ)
  int m_direction;

  //! number of owned quadrants
  const int32_t m_num_owned;

  //! number of ghost quadrants
  const int32_t m_num_ghosts;

  //! block sizes (no ghost)
  const block_size_t<dim> m_block_sizes;

  //! hydro settings (EOS parameters)
  HydroSettings m_hydro_settings;

  //! Equation of State (EoS) parameters
  EosWrapper_t<device_t> m_eos;

  //! volume fraction advection source term type (ALLAIRE_KOKH or KAPILA)
  core::VolFracAdvectionSourceTermType m_vol_frac_adv_source_term_type;

  //! time step
  real_t m_dt;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

public:
  /**
   * Perform time integration (MUSCL Godunov).
   *
   * \param[in]  time step (as computed by CFL condition)
   *
   */
  ReadFluxesAndConservativeUpdateFunctor(ConfigMap const &                  config_map,
                                         StencilHelper_t const &            stencil_helper,
                                         orchard_key_view_t const &         orchard_keys,
                                         conformal_status_view_type const & conformal_status,
                                         AMRMeshInfo const &                amr_mesh_info,
                                         DataArrayBlock_t const &           u_in,
                                         DataArrayBlock_t const &           u_out,
                                         DataArrayBlock_t const &           fluxes,
                                         DataArrayBlock_t const &           u_star,
                                         int                                direction,
                                         HydroSettings const &              hydro_settings,
                                         EosWrapper_t<device_t> const &     eos,
                                         real_t                             dt);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in a group of octant
  static void
  apply(ConfigMap const &                  config_map,
        amr_hashmap_t const &              amr_hashmap,
        orchard_key_view_t const &         orchard_keys,
        conformal_status_view_type const & conformal_status,
        AMRMeshInfo const &                amr_mesh_info,
        DataArrayBlock_t const &           Uin,
        DataArrayBlock_t const &           Uout,
        DataArrayBlock_t const &           fluxes,
        DataArrayBlock_t const &           u_star,
        int                                direction,
        brick_size_t<dim> const &          brick_sizes,
        Kokkos::Array<bool, dim> const &   is_brick_periodic,
        HydroSettings const &              hydro_settings,
        EosWrapper_t<device_t> const &     eos,
        real_t                             dt);

  // ====================================================================
  // ====================================================================
  /**
   * return true when octant is a "owned" octant (not a ghost or outside)
   */
  KOKKOS_INLINE_FUNCTION bool
  is_owned_quadrant(iOct_t const & iOct_global) const
  {
    return iOct_global < m_amr_mesh_info.local_num_quadrants();
  }

  // ====================================================================
  // ====================================================================
  /**
   * Get conservative variables state vector.
   *
   * \param[in] i,j identifies location in flux array
   * \param[in] iOct identifies a quadrant (owned + ghosts)
   *
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  read_flux(int32_t i, int32_t j, int32_t iOct) const
  {

    HydroState<2> flux;

    flux[Hydro::IAD0] = m_Fluxes(i, j, Hydro::IAD0, iOct);
    flux[Hydro::IAD1] = m_Fluxes(i, j, Hydro::IAD1, iOct);
    flux[Hydro::IA0] = m_Fluxes(i, j, Hydro::IA0, iOct);
    flux[Hydro::IA1] = m_Fluxes(i, j, Hydro::IA1, iOct);
    flux[Hydro::IP] = m_Fluxes(i, j, Hydro::IP, iOct);
    flux[Hydro::IU] = m_Fluxes(i, j, Hydro::IU, iOct);
    flux[Hydro::IV] = m_Fluxes(i, j, Hydro::IV, iOct);

    return flux;

  } // read_flux

  // ====================================================================
  // ====================================================================
  /**
   * Get conservative variables state vector.
   *
   * \param[in] i,j,k identifies in flux array
   * \param[in] iOct identifies a quadrant (owned + ghosts)
   *
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  read_flux(int32_t i, int32_t j, int32_t k, int32_t iOct) const
  {

    HydroState<3> flux;

    flux[Hydro::IAD0] = m_Fluxes(i, j, k, Hydro::IAD0, iOct);
    flux[Hydro::IAD1] = m_Fluxes(i, j, k, Hydro::IAD1, iOct);
    flux[Hydro::IA0] = m_Fluxes(i, j, k, Hydro::IA0, iOct);
    flux[Hydro::IA1] = m_Fluxes(i, j, k, Hydro::IA1, iOct);
    flux[Hydro::IP] = m_Fluxes(i, j, k, Hydro::IP, iOct);
    flux[Hydro::IU] = m_Fluxes(i, j, k, Hydro::IU, iOct);
    flux[Hydro::IV] = m_Fluxes(i, j, k, Hydro::IV, iOct);
    flux[Hydro::IW] = m_Fluxes(i, j, k, Hydro::IW, iOct);

    return flux;

  } // read_flux

  // ====================================================================
  // ====================================================================
  /**
   * Get flux from fine neighbor.
   *
   * \param[in] iOct_cur current octant id
   * \param[in] coords current cell cartesian coordinate (inside current block)
   * \param[in] shift indicates local direction to fine neighbor
   * \param[in] use_right_flux boolean value to tell if we want to use left or right flux
   *
   */
  KOKKOS_INLINE_FUNCTION auto
  get_flux_from_fine_neighbor(index_t const &      iOct_cur,
                              coord_t<dim> const & coords,
                              shift_t<dim> const & shift,
                              bool                 use_right_flux) const
  {
    const auto           key_cur = m_orchard_keys_device(iOct_cur);
    const CellLocation_t cell_loc{ coords, key_cur, iOct_cur, false };
    const auto           cell_loc_neigh = m_stencil_helper.getNeighLocFinerNearer(cell_loc, shift);

    HydroState<dim> flux;

    flux[Hydro::IAD0] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, Hydro::IAD0, m_Fluxes, use_right_flux);
    flux[Hydro::IAD1] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, Hydro::IAD1, m_Fluxes, use_right_flux);
    flux[Hydro::IA0] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, Hydro::IA0, m_Fluxes, use_right_flux);
    flux[Hydro::IA1] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, Hydro::IA1, m_Fluxes, use_right_flux);
    flux[Hydro::IP] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, Hydro::IP, m_Fluxes, use_right_flux);
    flux[Hydro::IU] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, Hydro::IU, m_Fluxes, use_right_flux);
    flux[Hydro::IV] = m_stencil_helper.compute_face_siblings_sum(
      cell_loc_neigh, Hydro::IV, m_Fluxes, use_right_flux);

    if constexpr (dim == 3)
    {
      flux[Hydro::IW] = m_stencil_helper.compute_face_siblings_sum(
        cell_loc_neigh, Hydro::IW, m_Fluxes, use_right_flux);
    }

    return flux;

  } // get_flux_from_fine_neighbor_flux

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION auto
  get_ustar_flux_from_fine_neighbor(index_t const &      iOct_cur,
                                    coord_t<dim> const & coords,
                                    shift_t<dim> const & shift,
                                    bool                 use_right_flux) const
  {
    const auto           key_cur = m_orchard_keys_device(iOct_cur);
    const CellLocation_t cell_loc{ coords, key_cur, iOct_cur, false };
    const auto           cell_loc_neigh = m_stencil_helper.getNeighLocFinerNearer(cell_loc, shift);

    return m_stencil_helper.compute_face_siblings_sum(cell_loc_neigh, 0, m_u_star, use_right_flux);

  } // get_ustar_flux_from_fine_neighbor

  // ====================================================================
  // ====================================================================
  /**
   * Update state vector (hydro variables only).
   *
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  update_U(int32_t i, int32_t j, int32_t iOct, HydroState<2> const & flux) const
  {
    constexpr real_t eps = KALYPSSO_NUM(1e-12);

    m_Uout(i, j, Hydro::IAD0, iOct) += flux[Hydro::IAD0];
    if (m_Uout(i, j, Hydro::IAD0, iOct) < eps)
      m_Uout(i, j, Hydro::IAD0, iOct) = ZERO_F;

    m_Uout(i, j, Hydro::IAD1, iOct) += flux[Hydro::IAD1];
    if (m_Uout(i, j, Hydro::IAD1, iOct) < eps)
      m_Uout(i, j, Hydro::IAD1, iOct) = ZERO_F;

    m_Uout(i, j, Hydro::IA0, iOct) += flux[Hydro::IA0];
    if (m_Uout(i, j, Hydro::IA0, iOct) < eps)
      m_Uout(i, j, Hydro::IA0, iOct) = ZERO_F;
    if (m_Uout(i, j, Hydro::IA0, iOct) > ONE_F - eps)
      m_Uout(i, j, Hydro::IA0, iOct) = ONE_F;

    m_Uout(i, j, Hydro::IA1, iOct) += flux[Hydro::IA1];
    if (m_Uout(i, j, Hydro::IA1, iOct) < eps)
      m_Uout(i, j, Hydro::IA1, iOct) = ZERO_F;
    if (m_Uout(i, j, Hydro::IA1, iOct) > ONE_F - eps)
      m_Uout(i, j, Hydro::IA1, iOct) = ONE_F;

    m_Uout(i, j, Hydro::IP, iOct) += flux[Hydro::IP];
    m_Uout(i, j, Hydro::IU, iOct) += flux[Hydro::IU];
    m_Uout(i, j, Hydro::IV, iOct) += flux[Hydro::IV];

  } // update_U - 2d

  // ====================================================================
  // ====================================================================
  /**
   * Update state vector (hydro variables only).
   *
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] k identifies location in the ghosted block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  update_U(int32_t i, int32_t j, int32_t k, int32_t iOct, HydroState<3> const & flux) const
  {
    constexpr real_t eps = KALYPSSO_NUM(1e-12);

    m_Uout(i, j, k, Hydro::IAD0, iOct) += flux[Hydro::IAD0];
    if (m_Uout(i, j, k, Hydro::IAD0, iOct) < eps)
      m_Uout(i, j, k, Hydro::IAD0, iOct) = ZERO_F;

    m_Uout(i, j, k, Hydro::IAD1, iOct) += flux[Hydro::IAD1];
    if (m_Uout(i, j, k, Hydro::IAD1, iOct) < eps)
      m_Uout(i, j, k, Hydro::IAD1, iOct) = ZERO_F;

    m_Uout(i, j, k, Hydro::IA0, iOct) += flux[Hydro::IA0];
    if (m_Uout(i, j, k, Hydro::IA0, iOct) < eps)
      m_Uout(i, j, k, Hydro::IA0, iOct) = ZERO_F;
    if (m_Uout(i, j, k, Hydro::IA0, iOct) > ONE_F - eps)
      m_Uout(i, j, k, Hydro::IA0, iOct) = ONE_F;

    m_Uout(i, j, k, Hydro::IA1, iOct) += flux[Hydro::IA1];
    if (m_Uout(i, j, k, Hydro::IA1, iOct) < eps)
      m_Uout(i, j, k, Hydro::IA1, iOct) = ZERO_F;
    if (m_Uout(i, j, k, Hydro::IA1, iOct) > ONE_F - eps)
      m_Uout(i, j, k, Hydro::IA1, iOct) = ONE_F;

    m_Uout(i, j, k, Hydro::IP, iOct) += flux[Hydro::IP];
    m_Uout(i, j, k, Hydro::IU, iOct) += flux[Hydro::IU];
    m_Uout(i, j, k, Hydro::IV, iOct) += flux[Hydro::IV];
    m_Uout(i, j, k, Hydro::IW, iOct) += flux[Hydro::IW];

  } // update_U - 3d

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  read_fluxes_and_update_2d(index_t const & cell_index, index_t const & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  read_fluxes_and_update_3d(const index_t & cell_index, const index_t & iOct_local) const;

  // ====================================================================
  // ====================================================================
  /**
   * Compute source term scaling factor to account for either Allaire-Kokh or Kapila model.
   */
  KOKKOS_INLINE_FUNCTION real_t
  get_source_term_factor(const index_t & cell_index, const index_t & iOct_local, size_t i_mat) const
  {
    if (m_vol_frac_adv_source_term_type == +core::VolFracAdvectionSourceTermType::KAPILA)
    {
      // recompute pressure in current cell

      // 1. cell-centered conservative variables in current cell
      const auto uLoc = get_conservative_variables(m_Uin, cell_index, iOct_local);

      // 2. recompute pressure
      const auto pressure = models::compute_mixture_pressure<dim>(uLoc, m_hydro_settings, m_eos);

      // 3. material bulk modulus
      const auto kappa0 =
        i_mat == 0 ? m_eos.material_bulk_modulus(0, pressure, uLoc[Hydro::IA0], uLoc[Hydro::IAD0])
                   : m_eos.material_bulk_modulus(1, pressure, uLoc[Hydro::IA1], uLoc[Hydro::IAD1]);

      // 4. mixture bulk modulus
      const auto kappa_bar = m_eos.mixture_bulk_modulus(uLoc[Hydro::ID],
                                                        pressure,
                                                        uLoc[Hydro::IA0],
                                                        uLoc[Hydro::IA1],
                                                        uLoc[Hydro::IAD0],
                                                        uLoc[Hydro::IAD1]);

      return (ONE_F - kappa_bar / kappa0);
    }

    // default value (Allaire-Kokh model)
    return ONE_F;
  }

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // ReadFluxesAndConservativeUpdateFunctor

// explicit template instantiation
extern template class ReadFluxesAndConservativeUpdateFunctor<2, kalypsso::DefaultDevice>;
extern template class ReadFluxesAndConservativeUpdateFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_READ_FLUXES_AND_CONSERVATIVE_UPDATE_FUNCTOR_H_
