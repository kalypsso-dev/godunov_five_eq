// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file solver_godunov_five_eq.cpp
 * \brief A two-fluid hydrodynamics solver using the 5 equations model.
 */
#include <cstdlib>
#include <cstdio>
#include <string>

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/real_type.h>   // choose between single and double precision
#include <kalypsso/core/HydroParams.h> // read parameter file

#ifdef KALYPSSO_CORE_USE_MPI
#  include <mpi.h>
#endif // KALYPSSO_CORE_USE_MPI

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/core/SolverBase.h>
#include <godunov_five_eq/SolverGodunovFiveEq.h>

#include <kalypsso/utils/config/ConfigMap.h>

#include <kalypsso/core/ComputeError.h>
#include <kalypsso/core/OutputParams.h>
#include <kalypsso/core/ComputeDataSliceAlongLine.h>

#include <kalypsso/core/cmdline_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>

// required for post-processing
#include <kalypsso/core/problems/TwoFluidShockTubeParams.h>

// banner
#include "kalypsso_core_version.h"
#include <kalypsso/core/kalypsso_core_git_info.h>
#include <kalypsso/core/kalypsso_core_build_info.h>

#ifdef KALYPSSO_CORE_USE_CPPTRACE
#  include <kalypsso/core/cpptrace_utils.h>
#endif // KALYPSSO_CORE_USE_CPPTRACE

namespace kalypsso
{

/* ============================================================ */
/* ============================================================ */
/* ============================================================ */
template <size_t dim, typename device_t>
void
run_simulation(ParallelEnv const &       par_env,
               ConfigMap const &         config_map,
               [[maybe_unused]] int &    argc,
               [[maybe_unused]] char **& argv)
{

  // test: create a HydroParams object
  HydroParams params = HydroParams(config_map);

  // initialize workspace memory (U, U2, ...)
  auto solver =
    godunov_five_eq::SolverGodunovFiveEq<dim, device_t>::create(par_env, params, config_map);

  // some monitoring prints
  {
    const auto eos_type = core::eos::get_eos_type(config_map);
    KALYPSSO_INFO("EOS type : {}", eos_type._to_string());
  }

  // diagnostics
  const auto conservativity_check_enabled =
    config_map.getBool("diagnostic", "conservativity_checks", true);

  // start computation
  if (par_env.rank() == 0)
  {
    KALYPSSO_INFO("Start computation....");
  }

  solver->profiling_mgr().get_whole_region().start();

  // register conservative integral values at initial time
  if (conservativity_check_enabled)
    solver->register_volume_integrals(true);

  // static AMR (froze AMR mesh computed in initial condition) ?
  const auto static_amr = config_map.getBool("amr", "static", false);
  if (static_amr)
  {
    solver->deactivate_amr_cycle();
  }

  // Hydrodynamics solver time loop
  solver->run();

  // register conservative integral values at final time
  if (conservativity_check_enabled)
  {
    solver->register_volume_integrals(false);
    solver->print_conservativity_check_report();
  }

  // save last time step as a regular checkpoint
  const auto output_params = OutputParams(config_map);
  if (output_params.nOutput != 0)
  {
    // save solution, this is not a pure checkpoint
    // just a regular output with all required fields for a checkpoint
    constexpr bool pure_checkpoint = false;
    solver->save_solution(pure_checkpoint);

    // save p4est mesh
    auto derived_solver =
      dynamic_cast<godunov_five_eq::SolverGodunovFiveEq<dim, device_t> *>(solver);
    const std::string mesh_filename = derived_solver->output_basename() + ".p4est";
    derived_solver->template save_p4est_mesh<dim>(mesh_filename,
                                                  derived_solver->amr_mesh()->forest());
  }

  // save records of mixing computation monitoring
  {
    auto solver_five_eq =
      dynamic_cast<godunov_five_eq::SolverGodunovFiveEq<dim, device_t> *>(solver);
    if (solver_five_eq->mixing_monitor().enabled())
    {
      // record last value
      solver_five_eq->mixing_monitor_record();
      solver_five_eq->mixing_monitor().save();
    }
  }

  solver->profiling_mgr().get_whole_region().stop();

  // =================================================================
  // here we do something specific to a given test case
  // =================================================================

  // =================================================================================
  // when doing a shock tube problem, dump a 1D slice of data
  // =================================================================================
  if (!solver->problem_name().compare("two_fluid_shock_tube"))
  {
    using FiveEq = godunov_five_eq::models::FiveEq;

    auto solver_five_eq =
      dynamic_cast<godunov_five_eq::SolverGodunovFiveEq<dim, device_t> *>(solver);
    // const auto st_params = TwoFluidShockTubeParams(config_map);
    const auto st_name = config_map.getString("two_fluid_shock_tube", "name", "shock_tube");

    const auto cell_var_ids = std::vector<int32_t>{
      solver_five_eq->model().get_fieldmap()[FiveEq::ID0],
      solver_five_eq->model().get_fieldmap()[FiveEq::ID1],
      solver_five_eq->model().get_fieldmap()[FiveEq::IE],
      solver_five_eq->model().get_fieldmap()[FiveEq::IU],
      solver_five_eq->model().get_fieldmap()[FiveEq::IPHI],
    };

    const auto cell_var_names = std::vector<std::string>{ "rho0", "rho1", "etot", "rhou", "phi" };

    kalypsso::core::ComputeDataSliceAlongLine<dim, device_t>::apply(
      solver_five_eq->U(),
      0,
      solver_five_eq->mesh_map()->get_amr_mesh_info().local_num_quadrants(),
      IX,
      solver_five_eq->mesh_map()->orchard_keys(),
      cell_var_ids,
      cell_var_names,
      st_name,
      par_env,
      config_map);

  } // shock tube post-processing

  if (!solver->problem_name().compare("static_droplet"))
  {
    auto solver_impl = dynamic_cast<godunov_five_eq::SolverGodunovFiveEq<dim, device_t> *>(solver);

    const auto radius = config_map.getReal("static_droplet", "radius", KALYPSSO_NUM(1.0));

    using exec_space = typename device_t::execution_space;
    const auto num_quadrants = solver_impl->U().num_quadrants();
    const auto num_cells = solver_impl->U().num_cells();
    const auto Udata = solver_impl->U();
    const auto fm = solver_impl->model().get_fieldmap();
    const auto curvature = solver_impl->smooth_interface_function_data().m_curvature;

    real_t              sum_curvature, total_sum_curvature = ZERO_F;
    Kokkos::Sum<real_t> reducer(sum_curvature);

    Kokkos::parallel_reduce(
      "Compute sum of curvature",
      Kokkos::RangePolicy<exec_space>(0, num_quadrants * num_cells),
      KOKKOS_LAMBDA(const int32_t index, real_t & local_sum) {
        const auto iOct = index / Udata.num_cells();
        const auto cellindex = index - iOct * num_cells;

        const auto phi = Udata(cellindex, fm[godunov_five_eq::models::FiveEq::IPHI], iOct);
        const bool is_near_interface = phi * (ONE_F - phi) > KALYPSSO_NUM(0.001);


        const auto ijk = cellindex_to_coord<dim>(cellindex, Udata.block_size());
        const auto curv = curvature(ijk, 0, iOct);

        if (is_near_interface)
        {
          local_sum += (curv - ONE_F / radius) * (curv - ONE_F / radius);
        }
      },
      reducer);

    total_sum_curvature = sum_curvature;
#ifdef KALYPSSO_CORE_USE_MPI
    par_env.comm().template MPI_Allreduce<MpiComm::SUM>(&sum_curvature, &total_sum_curvature, 1);
#endif // KALYPSSO_CORE_USE_MPI

    int32_t              sum_locations, total_sum_locations = 0;
    Kokkos::Sum<int32_t> reducer2(sum_locations);

    Kokkos::parallel_reduce(
      "Compute sum of curvature",
      Kokkos::RangePolicy<exec_space>(0, num_quadrants * num_cells),
      KOKKOS_LAMBDA(const int32_t index, int32_t & local_sum) {
        const auto iOct = index / num_cells;
        const auto cellindex = index - iOct * Udata.num_cells();

        const auto phi = Udata(cellindex, fm[godunov_five_eq::models::FiveEq::IPHI], iOct);
        const auto is_near_interface = phi * (ONE_F - phi) > KALYPSSO_NUM(0.001);

        if (is_near_interface)
          local_sum++;
      },
      reducer2);

    total_sum_locations = sum_locations;
#ifdef KALYPSSO_CORE_USE_MPI
    par_env.comm().template MPI_Allreduce<MpiComm::SUM>(&sum_locations, &total_sum_locations, 1);
#endif // KALYPSSO_CORE_USE_MPI

    KALYPSSO_INFO("Static droplet L2 error on curvature is {:010.8f}\n",
                  sqrt(total_sum_curvature / static_cast<real_t>(total_sum_locations)));
  }

  KALYPSSO_INFO("final time is {:010.2f}\n", solver->current_time());

  solver->print_monitoring_info_final();

  delete solver;

} // run_simulation

} // namespace kalypsso

// ===============================================================
// ===============================================================
// ===============================================================
int
main(int argc, char * argv[])
{

  {
    // create parallel environment (p4est, MPI, kokkos, ...)
    kalypsso::ParallelEnv par_env(argc, argv);

#ifdef KALYPSSO_CORE_USE_SPDLOG
    // logger setup
    kalypsso::kalypsso_spdlog_config(argc, argv, par_env.rank(), par_env.size());
#endif

#ifdef KALYPSSO_CORE_USE_CPPTRACE
    kalypsso::cpptrace_initialize();
#endif // KALYPSSO_CORE_USE_CPPTRACE

    // parse command line arguments
    if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--version"))
    {
      if (par_env.rank() == 0)
      {
        kalypsso::GitRevisionInfo::print();
        kalypsso::BuildInfo::print();
      }
      return EXIT_SUCCESS;
    }
    else if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--help"))
    {
      if (par_env.rank() == 0)
      {
        // clang-format off
        std::cout << "Example cmdline: \"mpirun -np 1 ./solver_godunov_five_eq --ini test_two_fluid_shock_tube_2d.ini\"\n";
        // clang-format on
      }
      return EXIT_SUCCESS;
    }

    // print kalypsso banner
    if (par_env.rank() == 0)
    {
      kalypsso::GitRevisionInfo::print();
      kalypsso::BuildInfo::print();
    }

    // check if user passed a custom ini filename
    // provide a default input filename if user did'nt set one
    std::string input_filename = kalypsso::cmdline_get_string(argv, argv + argc, "--ini");

    if (input_filename.size() == 0)
      input_filename = "test_two_fluid_shock_tube_2d.ini";

    // only MPI rank 0 actually reads input file, and broadcast it to all other MPI processor
    kalypsso::ConfigMap config_map = kalypsso::broadcast_parameters(input_filename);

    const auto dim = kalypsso::get_dim(config_map);
    assertm(dim == 2 or dim == 3, "[solver_godunov_five_eq] Wrong dimension");

    // run some test
    if (dim == 2)
    {
      kalypsso::run_simulation<2, kalypsso::DefaultDevice>(par_env, config_map, argc, argv);
    }
    else if (dim == 3)
    {
      kalypsso::run_simulation<3, kalypsso::DefaultDevice>(par_env, config_map, argc, argv);
    }
  }

  return EXIT_SUCCESS;

} // end main
