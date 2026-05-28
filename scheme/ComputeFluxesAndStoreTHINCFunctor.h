// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeFluxesAndStoreTHINCFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_COMPUTE_FLUXES_AND_STORE_THINC_FUNCTOR_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_COMPUTE_FLUXES_AND_STORE_THINC_FUNCTOR_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>

#include <kalypsso/core/ConformalFaceStatus.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/TimeIntegratorConfig.h>
#include <kalypsso/core/THINCParams.h>

#include <godunov_five_eq/models/HydroState.h>
#include <godunov_five_eq/models/utils.h>
#include <godunov_five_eq/models/RiemannSolvers.h>

#include <type_traits>

namespace kalypsso
{

namespace godunov_five_eq
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute fluxes (on conservative variables) and store.
 *
 * - use THINC (Tangent Hyperbolic INterface Capture) for reconstructing states on cell interface.
 * - (optional) add surface tension is incorporated using CSF (Continuum Surface Force) model, a
 * modified version of HLLC is used.
 *
 * For THINC reconstruction see the following reference and reference therein:
 * Zhang et al., A hybrid WENO5IS-THINC reconstruction scheme for compressible multiphase flows, JCP
 * Vol 498 (2024), 112672. https://doi.org/10.1016/j.jcp.2023.112672
 *
 * For the modified HLLC Riemann solver see
 * Garrick et al., A finite-volume HLLC-based scheme for compressible interfacial flows with surface
 * tension, JCP 339 (2017), 46-67. https://doi.org/10.1016/j.jcp.2017.03.007
 *
 * \sa ComputeFluxesAndStoreFunctor
 *
 */
template <size_t dim, typename device_t>
class ComputeFluxesAndStoreTHINCFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int64_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using Hydro = models::FiveEq<dim>;

  // access quadrant <-> orchard key (hence AMR level and quadrant size)
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, int>;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! fluxes (output)
  DataArrayBlock_t m_Fluxes;

  //! star velocity (output)
  DataArrayBlock_t m_u_star;

  //! a ghosted block array of primitive variables (ghost width is 2)
  //! size :
  //! if implem version 0 : owned + ghost quadrants
  //! if implem version 1 : size of group of quadrants
  DataArrayGhostedBlock_t m_q;

  //! ghosted block data arrays (ghost width is 1) - slopes along X
  DataArrayGhostedBlock_t m_slopes_x;

  //! ghosted block data arrays (ghost width is 1) - slopes along Y
  DataArrayGhostedBlock_t m_slopes_y;

  //! ghosted block data arrays (ghost width is 1) - slopes along Z - only used when dim=3
  DataArrayGhostedBlock_t m_slopes_z;

  //! offset to first octant in flux array where to write
  const int32_t m_iOct_flux_offset;

  //! number of quadrants to process
  const int32_t m_num_quads;

  //! flux direction (IX, IY or IZ)
  int m_direction;

  //! block sizes (no ghost)
  const block_size_t<dim> m_block_sizes;

  //! number of cells per leaf block
  const int32_t m_nbCellsPerLeaf;

  //! hydro settings (EOS parameters)
  HydroSettings m_hydro_settings;

  //! EOS parameters
  EosWrapper_t<device_t> m_eos;

  //! time step
  real_t m_dt;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! time integrator id
  const TimeIntegrator m_time_integrator;

  //! thinc beta
  const THINCParams m_thinc;

public:
  /**
   * Compute Godunov fluxes along a given direction.
   *
   * \param[in]  time step (as computed by CFL condition)
   *
   */
  ComputeFluxesAndStoreTHINCFunctor(orchard_key_view_t const &      orchard_keys,
                                    AMRMeshInfo const &             amr_mesh_info,
                                    DataArrayBlock_t const &        fluxes,
                                    DataArrayBlock_t const &        u_star,
                                    DataArrayGhostedBlock_t const & q_ghosted,
                                    DataArrayGhostedBlock_t const & slopes_x,
                                    DataArrayGhostedBlock_t const & slopes_y,
                                    DataArrayGhostedBlock_t const & slopes_z,
                                    int32_t                         iOct_flux_offset,
                                    int32_t                         num_quads,
                                    int                             direction,
                                    HydroSettings const &           hydro_settings,
                                    EosWrapper_t<device_t> const &  eos,
                                    real_t                          dt,
                                    real_t                          scaling_factor,
                                    TimeIntegrator                  time_integrator,
                                    THINCParams const &             thinc_params);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  static void
  apply(ConfigMap const &               config_map,
        orchard_key_view_t const &      orchard_keys,
        AMRMeshInfo const &             amr_mesh_info,
        DataArrayBlock_t const &        fluxes,
        DataArrayBlock_t const &        u_star,
        DataArrayGhostedBlock_t const & q_ghosted,
        DataArrayGhostedBlock_t const & slopes_x,
        DataArrayGhostedBlock_t const & slopes_y,
        DataArrayGhostedBlock_t const & slopes_z,
        int32_t                         iOct_flux_offset,
        int32_t                         num_quads,
        int                             direction,
        HydroSettings const &           hydro_settings,
        EosWrapper_t<device_t> const &  eos,
        real_t                          dt);

  // ====================================================================
  // ====================================================================
  /**
   * Get primitive variables state vector.
   *
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (from 0 to owned+ghost-1)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_prim_variables(int32_t i, int32_t j, int32_t iOct_local) const
  {

    HydroState<2> q;

    q[Hydro::IAD0] = m_q(i, j, Hydro::IAD0, iOct_local);
    q[Hydro::IAD1] = m_q(i, j, Hydro::IAD1, iOct_local);
    q[Hydro::IA0] = m_q(i, j, Hydro::IA0, iOct_local);
    q[Hydro::IP] = m_q(i, j, Hydro::IP, iOct_local);
    q[Hydro::IU] = m_q(i, j, Hydro::IU, iOct_local);
    q[Hydro::IV] = m_q(i, j, Hydro::IV, iOct_local);

    return q;

  } // get_prim_variables

  // ====================================================================
  // ====================================================================
  /**
   * Get primitive variables state vector.
   *
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (from 0 to owned+ghost-1)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_prim_variables(int32_t i, int32_t j, int32_t k, int32_t iOct_local) const
  {

    HydroState<3> q;

    q[Hydro::IAD0] = m_q(i, j, k, Hydro::IAD0, iOct_local);
    q[Hydro::IAD1] = m_q(i, j, k, Hydro::IAD1, iOct_local);
    q[Hydro::IA0] = m_q(i, j, k, Hydro::IA0, iOct_local);
    q[Hydro::IP] = m_q(i, j, k, Hydro::IP, iOct_local);
    q[Hydro::IU] = m_q(i, j, k, Hydro::IU, iOct_local);
    q[Hydro::IV] = m_q(i, j, k, Hydro::IV, iOct_local);
    q[Hydro::IW] = m_q(i, j, k, Hydro::IW, iOct_local);

    return q;

  } // get_prim_variables

  // ====================================================================
  // ====================================================================
  /**
   * Set flux (hydro variables only).
   *
   * \param[in] i identifies location in the flux in block
   * \param[in] j identifies location in the flux in block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  set_flux(int32_t i, int32_t j, int32_t iOct, HydroState<2> const & flux, real_t ustar_flux) const
  {
    iOct += m_iOct_flux_offset;

    m_Fluxes(i, j, Hydro::IAD0, iOct) = flux[Hydro::IAD0];
    m_Fluxes(i, j, Hydro::IAD1, iOct) = flux[Hydro::IAD1];
    m_Fluxes(i, j, Hydro::IA0, iOct) = flux[Hydro::IA0];

    m_Fluxes(i, j, Hydro::IP, iOct) = flux[Hydro::IP];
    m_Fluxes(i, j, Hydro::IU, iOct) = flux[Hydro::IU];
    m_Fluxes(i, j, Hydro::IV, iOct) = flux[Hydro::IV];

    m_u_star(i, j, 0, iOct) = ustar_flux;

  } // set_flux - 2d

  // ====================================================================
  // ====================================================================
  /**
   * Set flux (hydro variables only).
   *
   * \param[in] i identifies location in the flux in block
   * \param[in] j identifies location in the flux in block
   * \param[in] k identifies location in the flux in block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   *
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  set_flux(int32_t               i,
           int32_t               j,
           int32_t               k,
           int32_t               iOct,
           HydroState<3> const & flux,
           real_t                ustar_flux) const
  {
    iOct += m_iOct_flux_offset;

    m_Fluxes(i, j, k, Hydro::IAD0, iOct) = flux[Hydro::IAD0];
    m_Fluxes(i, j, k, Hydro::IAD1, iOct) = flux[Hydro::IAD1];
    m_Fluxes(i, j, k, Hydro::IA0, iOct) = flux[Hydro::IA0];

    m_Fluxes(i, j, k, Hydro::IP, iOct) = flux[Hydro::IP];
    m_Fluxes(i, j, k, Hydro::IU, iOct) = flux[Hydro::IU];
    m_Fluxes(i, j, k, Hydro::IV, iOct) = flux[Hydro::IV];
    m_Fluxes(i, j, k, Hydro::IW, iOct) = flux[Hydro::IW];

    m_u_star(i, j, k, 0, iOct) = ustar_flux;

  } // set_flux - 3d

  // ====================================================================
  // ====================================================================
  /**
   * Reconstruct an hydro state at a cell border location specified by offsets.
   *
   * This is equivalent to trace operation in Ramses.
   * We just extrapolate primitive variables (at cell center) to border
   * using limited slopes.
   *
   * \note offsets are given in units dx/2, i.e. a vector containing only 1.0 or -1.0
   *
   * \param[in] q primitive variables at cell center
   * \param[in] i_s X coordinate to access slope array
   * \param[in] j_s Y coordinate to access slope array
   * \param[in] iOct_local index to octant in local array
   * \param[in] offsets identifies where to reconstruct
   * \param[in] dtdx dt divided by dx (only required for Hancock)
   * \param[in] dtdy dt divided by dy (only required for Hancock)
   *
   * \return qr reconstructed state (primitive variables)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  reconstruct_state_2d(const HydroState<2> & q,
                       int32_t               i_s,
                       int32_t               j_s,
                       int32_t               iOct_local,
                       const offsets_t<2> &  offsets,
                       real_t                dtdx,
                       real_t                dtdy) const;

  // ====================================================================
  // ====================================================================
  /**
   * Reconstruct an hydro state at a cell border location specified by offsets (3d version).
   *
   * This is equivalent to trace operation in Ramses.
   * We just extrapolate primitive variables (at cell center) to border
   * using limited slopes.
   *
   * \note offsets are given in units dx/2, i.e. a vector containing only 1.0 or -1.0
   *
   * \param[in] q primitive variables at cell center
   * \param[in] is X coordinate to access slope array
   * \param[in] js Y coordinate to access slope array
   * \param[in] ks Y coordinate to access slope array
   * \param[in] iOct_local index to octant in local array
   * \param[in] offsets identifies where to reconstruct
   * \param[in] dtdx dt divided by dx
   * \param[in] dtdy dt divided by dy
   * \param[in] dtdz dt divided by dz
   *
   * \return qr reconstructed state (primitive variables)
   *
   * \sa reconstruct_state_2d
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  reconstruct_state_3d(const HydroState<3> & q,
                       int32_t               is,
                       int32_t               js,
                       int32_t               ks,
                       int32_t               iOct_local,
                       const offsets_t<3> &  offsets,
                       real_t                dtdx,
                       real_t                dtdy,
                       real_t                dtdz) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_store_2d(int32_t const & cell_index, int32_t const & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_store_3d(const int32_t & cell_index, const int32_t & iOct_local) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // ComputeFluxesAndStoreTHINCFunctor

// explicit template instantiation
extern template class ComputeFluxesAndStoreTHINCFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeFluxesAndStoreTHINCFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_COMPUTE_FLUXES_AND_STORE_THINC_FUNCTOR_H_
