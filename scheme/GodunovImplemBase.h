// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GodunovImplemBase.h
 *
 * Godunov time integration implementation detail interface definition.
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_GODUNOV_IMPLEM_BASE_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_GODUNOV_IMPLEM_BASE_H_

// shared
#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_HDF5, ...
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h>
#include <kalypsso/core/HydroParams.h>
#include <kalypsso/core/AMRmesh.h>
#include <kalypsso/core/MeshMap.h>
#include <kalypsso/core/config_utils.h> // for get_block_sizes

// for surface tension support (should it be included in GodunovImplemBase ?)
#include <kalypsso/core/InterfaceNormalVectorAlgorithmParams.h>
#include <kalypsso/core/SmoothInterfaceFunctionData.h>
#include <kalypsso/core/THINCParams.h>

#include <godunov_five_eq/models/FiveEq.h>
#include <godunov_five_eq/eos/eos_utils.h>
#include <kalypsso/core/models/HydroSettings.h>

// for IO
#ifdef KALYPSSO_CORE_USE_HDF5
#  include <kalypsso/core/HDF5_Xdmf_Writer.h>
#endif

#include <kalypsso/utils/monitoring/ProfilingManager.h>

// profiling colors
#include <godunov_five_eq/profiling.h>

// AMR services
#include <kalypsso/utils/mpi/ParallelEnv.h>

#ifdef KALYPSSO_CORE_USE_MPI
#  include <kalypsso/core/MeshGhostsExchanger.h>
#endif // KALYPSSO_CORE_USE_MPI

// #include <godunov_five_eq/AddGravitySourceTerm.h>

#include <memory> // std::shared_ptr

namespace kalypsso
{
namespace godunov_five_eq
{

/**
 * Virtual base class for an actual Godunov implementation solver.
 */
template <size_t dim, typename device_t>
class GodunovImplemBase
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayBlockHost_t = DataArrayBlock<dim, real_t, HostDevice>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

#ifdef KALYPSSO_CORE_USE_HDF5
  using HDF5_Xdmf_Writer_t = HDF5_Xdmf_Writer<dim, device_t>;
#endif

  GodunovImplemBase(ParallelEnv const &      par_env,
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
    : m_par_env(par_env)
    , m_params(params)
    , m_hydro_settings(config_map)
    , m_config_map(config_map)
    , m_block_sizes(get_block_sizes<dim>(config_map))
    , m_brick_sizes(get_brick_sizes<dim>(config_map))
    , m_is_brick_periodic(get_brick_periodicity<dim>(config_map))
    , m_problem_name(config_map.getString("hydro", "problem", "unknown"))
    , m_profiling_mgr(profiling_manager)
    , m_amr_mesh(amr_mesh)
    , m_mesh_map(mesh_map)
#ifdef KALYPSSO_CORE_USE_MPI
    , m_mesh_ghosts_exchanger(mesh_ghosts_exchanger)
#endif // KALYPSSO_CORE_USE_MPI
    , m_eos(config_map)
    , m_thinc_params(config_map)
    , m_sifd(par_env, config_map, amr_mesh, mesh_map)
  {}

  virtual ~GodunovImplemBase() = default;

  //! resize only auxiliary data array (implementation dependent)
  virtual void
  resize_auxiliary_data() = 0;

  //! memory footprint monitoring
  virtual uint64_t
  total_mem_size_in_bytes() = 0;

  //! Perform actual time integration.
  virtual void
  do_time_step(DataArrayBlock_t U, DataArrayBlock_t U2, real_t dt) = 0;

  //! Perform actual time integration.
  virtual void
  do_time_step(DataArrayBlock_t U, DataArrayBlock_t U2, real_t t, real_t dt) = 0;

  // =====================================================================
  // =====================================================================
  //! fills ghost octants with primitive variables from MPI exchange
  void
  mpi_exchange_mirrors_and_ghosts([[maybe_unused]] DataArrayGhostedBlock_t q_ghosted_mg)
  {

    // This fence ensure that buffer q_ghosted_mg (output of ConvertToPrimitivesVariablesFunctor
    // device kernel) is ready (ie. kernel has finished) before MPI exchange actually starts.
    // Note that without this fence, mesh_ghosts_echanger may start too early to send data.
    Kokkos::fence();

#ifdef KALYPSSO_CORE_USE_MPI
    KALYPSSO_PROFILING_REGION_DEVICE(m_profiling_mgr, NUM_SCHEME_EXCHANGE_Q_MIRROR_GHOST);
    this->m_mesh_ghosts_exchanger.exchange_inplace(q_ghosted_mg);
#endif // KALYPSSO_CORE_USE_MPI

  } // mpi_exchange_mirrors_and_ghosts

  // =====================================================================
  // =====================================================================
  //! add gravity source term
  virtual void
  add_gravity_source_term([[maybe_unused]] DataArrayBlock_t u_in,
                          [[maybe_unused]] DataArrayBlock_t u_out,
                          [[maybe_unused]] real_t           dt)
  {

    KALYPSSO_PROFILING_REGION_DEVICE(m_profiling_mgr, NUM_SCHEME_GRAVITY);

    KALYPSSO_WARN("Please implement gravity for FiveEq !");
    // AddGravitySourceTerm<dim, device_t>::apply(
    //   this->m_config_map,
    //   u_in,
    //   u_out,
    //   fm,
    //   this->m_mesh_map.get_amr_mesh_info().local_num_quadrants(),
    //   HALF_F * dt);

  } // add_gravity_source_term

  //! Save data specific to the actual Godunov implementation.
  //! \return total number of bytes written
#ifdef KALYPSSO_CORE_USE_HDF5
  virtual uint64_t
  save_data(std::shared_ptr<HDF5_Xdmf_Writer_t> hdf5_writer) = 0;
#endif // KALYPSSO_CORE_USE_HDF5

  /* useful data provided by solver */
  //! Parallel environment.
  ParallelEnv const & m_par_env;

  //! hydrodynamics parameters settings
  HydroParams const & m_params;

  //! Hydro settings
  const HydroSettings m_hydro_settings;

  //! unordered map of parameters read from input ini file
  ConfigMap const & m_config_map;

  //! block sizes
  const block_size_t<dim> m_block_sizes;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_sizes;

  //! array of bool to tell if mesh is periodic or not
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! init condition name (or problem)
  std::string m_problem_name;

  //! Profiling manager
  ProfilingManager & m_profiling_mgr;

  //! AMR mesh for accessing mesh sizes (number of owned, mirror, ghost, outside quadrants)
  AMRmesh<dim> const & m_amr_mesh;

  //! mesh map is a helper class for accessing orchard keys
  MeshMap<dim, device_t> & m_mesh_map;

#ifdef KALYPSSO_CORE_USE_MPI
  //! MPI communications to exchange ghost block userdata
  MeshGhostsExchanger<dim, real_t, device_t> & m_mesh_ghosts_exchanger;
#endif // KALYPSSO_CORE_USE_MPI

  //! Equation of state wrapper
  EosWrapper_t<device_t> m_eos;

  //! THINC (interface reconstruction)
  THINCParams m_thinc_params;

  //! Smooth Interface Function (SMI) for THINC and surface tension support (optional).
  //! SMI activated when "smooth_interface_function"/"enabled" is true in input parameter file.
  SmoothInterfaceFunctionData<dim, device_t> m_sifd;

}; // class GodunovImplemBase

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_GODUNOV_IMPLEM_BASE_H_
