// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GodunovImplemV0.h
 *
 * Godunov time integration implementation detail version v0.
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_GODUNOV_IMPLEM_V0_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_GODUNOV_IMPLEM_V0_H_

#include <godunov_five_eq/scheme/GodunovImplemBase.h>

#include <kalypsso/core/utils_block.h>
#include <kalypsso/core/config_utils.h> // for get_block_sizes

#include <godunov_five_eq/models/HydroState.h>

// for surface tension support (should it be included in GodunovImplemBase ?)
#include <kalypsso/core/SmoothInterfaceFunctionData.h>

namespace kalypsso
{
namespace godunov_five_eq
{
/**
 * \class GodunovImplemV0
 *
 * version 0 features:
 *
 * - do compute fluxes and store them;
 * - in a separate kokkos functor perform update  (no atomic memory operations required),
 *  - the sub-domain decomposition is a bit more complex to do (so it is not currently
 *    implemented here);
 * - all intermediate ghosted block array must be sized upon the total number of
 *   local quadrants (could be interesting to monitor memory footprint during the run
 *   and compared with implem version 1).
 *
 * \sa GodunovImplemV0
 */
template <size_t dim, typename device_t>
class GodunovImplemV0 : public GodunovImplemBase<dim, device_t>
{
public:
  using GodunovImplemBase_t = GodunovImplemBase<dim, device_t>;
  using DataArrayBlock_t = typename GodunovImplemBase_t::DataArrayBlock_t;
  using DataArrayGhostedBlock_t = typename GodunovImplemBase_t::DataArrayGhostedBlock_t;

#ifdef KALYPSSO_CORE_USE_HDF5
  using HDF5_Xdmf_Writer_t = typename GodunovImplemBase_t::HDF5_Xdmf_Writer_t;
#endif

  GodunovImplemV0(ParallelEnv const &      par_env,
                  HydroParams const &      params,
                  ConfigMap const &        config_map,
                  ProfilingManager &       profiling_manager,
                  AMRmesh<dim> &           amr_mesh,
                  MeshMap<dim, device_t> & mesh_map
#ifdef KALYPSSO_CORE_USE_MPI
                  ,
                  MeshGhostsExchanger<dim, real_t, device_t> & mesh_ghosts_exchanger
#endif // KALYPSSO_CORE_USE_MPI
                  )
    : GodunovImplemBase_t(par_env,
                          params,
                          config_map,
                          profiling_manager,
                          amr_mesh,
                          mesh_map
#ifdef KALYPSSO_CORE_USE_MPI
                          ,
                          mesh_ghosts_exchanger
#endif // KALYPSSO_CORE_USE_MPI
                          )
    , m_Q_ghosted(this->m_block_sizes,
                  this->m_block_sizes + 2 * 2,
                  get_shift<dim>(-2),
                  "Q_ghosted (owned and ghosts)",
                  models::FiveEq<dim>::nbvar(),
                  0)
    , m_Q_ghosted_mg(this->m_block_sizes,
                     this->m_block_sizes + 2 * 2,
                     get_shift<dim>(-2),
                     "Q_ghosted_mg",
                     models::FiveEq<dim>::nbvar(),
                     0)
    , m_Slopes_x(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_x_group",
                 models::FiveEq<dim>::nbvar(),
                 0)
    , m_Slopes_y(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_y_group",
                 models::FiveEq<dim>::nbvar(),
                 0)
    , m_Slopes_z(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_z_group",
                 models::FiveEq<dim>::nbvar(),
                 0)
    , m_Fluxes("Fluxes",
               get_flux_block_sizes<dim>(this->m_block_sizes),
               models::FiveEq<dim>::nbvar(),
               0)
    , m_u_star("u_star", get_flux_block_sizes<dim>(this->m_block_sizes), 1, 0)
  {} // GodunovImplemV0

  // destructor
  ~GodunovImplemV0() = default;

  /**
   * Static creation method called by the solver factory.
   */
  static GodunovImplemBase_t *
  create(ParallelEnv const &      par_env,
         HydroParams const &      params,
         ConfigMap const &        config_map,
         ProfilingManager &       profiling_manager,
         AMRmesh<dim> &           amr_mesh,
         MeshMap<dim, device_t> & mesh_map
#ifdef KALYPSSO_CORE_USE_MPI
         ,
         MeshGhostsExchanger<dim, real_t, device_t> & mesh_ghosts_exchanger
#endif // KALYPSSO_CORE_USE_MPI
  )
  {
#ifdef KALYPSSO_CORE_USE_MPI
    GodunovImplemV0<dim, device_t> * impl = new GodunovImplemV0<dim, device_t>(
      par_env, params, config_map, profiling_manager, amr_mesh, mesh_map, mesh_ghosts_exchanger);
#else
    GodunovImplemV0<dim, device_t> * impl = new GodunovImplemV0<dim, device_t>(
      par_env, params, config_map, profiling_manager, amr_mesh, mesh_map);
#endif // KALYPSSO_CORE_USE_MPI

    return impl;
  }

  /**
   * Public interface
   */

  //! resize only auxiliary data array (implementation dependent)
  void
  resize_auxiliary_data() override;

  //! memory footprint monitoring
  uint64_t
  total_mem_size_in_bytes() override;

  //! Perform actual time integration.
  void
  do_time_step(DataArrayBlock_t U, DataArrayBlock_t U2, real_t dt) override;

  //! Perform actual time integration.
  void
  do_time_step(DataArrayBlock_t U, DataArrayBlock_t U2, real_t t, real_t dt) override;

#ifdef KALYPSSO_CORE_USE_HDF5
  //! Save internal data (useful for debug).
  //! this routine is intended to be called with a hdf5 writer "well" configured (with mesh, file
  //! name, etc...) typical use: from inside SolverGodunovFiveEq::save_solution_hdf5
  //!
  //! \return total number of bytes written
  uint64_t
  save_data(std::shared_ptr<HDF5_Xdmf_Writer_t> hdf5_writer) override;
#endif // KALYPSSO_CORE_USE_HDF5

private:
  /*
   * ghosted block arrays used for piece wise computation
   */
  //! hydrodynamics primitive - ghosted block - number of octants : owned + ghost
  //! ghostwidth of 2
  DataArrayGhostedBlock_t m_Q_ghosted;

  //! hydrodynamics primitive - ghosted block - number of octants: MPI mirrors + MPI ghosts
  //! \note the suffix _mg is meant to tell the developer that this array "lives" on
  //! mirror (m) and ghosts(g)
  //! ghostwidth of 2
  DataArrayGhostedBlock_t m_Q_ghosted_mg;

  //! slopes along X dir - ghosted block array of octant's block data - owned + ghosts
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_x;

  //! slopes along Y dir - ghosted block array of octant's block data - owned + ghosts
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_y;

  //! slopes along Z dir - ghosted block array of octant's block data - owned + ghosts
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_z;

  //! Temporary buffer used to stored hydrodynamics flux (owned + ghost + outside block).
  //! It will be reshaped as needed to store either fluxes along X, Y or Z direction.
  //! Fluxes is allocated with one extra field to store "ustar" necessary to compute
  //! the source term in the RHS of volumic fraction advection equation.
  DataArrayBlock_t m_Fluxes;

  //! Temporary buffer to store u_star (as computed by the Riemann solver).
  //! u_star is necessary to compute the source term in the RHS of volumic fraction
  //! advection equation.
  DataArrayBlock_t m_u_star;

  //! Convert conservative variables to primitive variables in mirror quadrants.
  //! Fills m_Q_ghosted_mg
  //!
  //! \param[in] U conservative variables array (owned + MPI ghost + outside quadrants)
  void
  convert_to_primitives_in_mirror_quads(DataArrayBlock_t U);

  //! Convert conservative variables to primitive variables in owned octants + copy ghost octants
  //!
  //! \param[in] U conservative variables (owned + MPI ghost + outside quadrants)
  void
  convert_to_primitives(DataArrayBlock_t U);

  //! compute limited slopes in owned and ghosts quadrants
  void
  compute_limited_slopes_in_owned_and_ghosts();

  //! Compute hydro fluxes in all (owned and ghosts) quandrants.
  void
  compute_fluxes_and_store_in_owned_and_ghosts(real_t dt, int direction);

  //! Update conservative variable in owned quadrants.
  void
  read_fluxes_and_update_in_owned(DataArrayBlock_t u_in,
                                  DataArrayBlock_t u_out,
                                  real_t           dt,
                                  int              direction);

}; // class GodunovImplemV0

extern template class GodunovImplemV0<2, kalypsso::DefaultDevice>;
extern template class GodunovImplemV0<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_GODUNOV_IMPLEM_V0_H_
