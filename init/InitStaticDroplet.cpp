// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitStaticDroplet.cpp
 */

#include <godunov_five_eq/init/InitStaticDroplet.h>
#include <godunov_five_eq/SolverGodunovFiveEq.h>

#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{
namespace godunov_five_eq
{

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitStaticDropletDataFunctor<dim, device_t>::apply(
  DataArrayBlock_t                     Udata,
  FieldMap<models::FiveEq>             fm,
  orchard_key_view_t<device_t>         orchard_keys,
  int32_t                              local_num_octants,
  InitialStates<dim, device_t> const & initial_states,
  ConfigMap const &                    config_map)
{
  // data init functor
  InitStaticDropletDataFunctor functor(
    Udata, fm, orchard_keys, local_num_octants, initial_states, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_five_eq::InitStaticDropletDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitStaticDropletDataFunctor::apply

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitStaticDropletDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  // makes enum Hydro::VarId available
  using Hydro = models::FiveEq;

  const auto & block_sizes = m_Udata.block_size();

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // cell size
  const auto dx =
    compute_cell_length<dim>(static_cast<uint8_t>(m_level_max), block_sizes[IX]) * m_scaling_factor;

  auto const & droplet_x = m_staticDropletParams.x;
  auto const & droplet_y = m_staticDropletParams.y;
  auto const & droplet_z = m_staticDropletParams.z;
  auto const & Delta = m_staticDropletParams.Delta;
  auto const & epsilon_coef = m_staticDropletParams.epsilon_coef;

  auto const & x = xyz[IX];
  auto const & y = xyz[IY];

  auto r2 = (x - droplet_x) * (x - droplet_x) + (y - droplet_y) * (y - droplet_y);
  if constexpr (dim == 3)
  {
    auto const & z = xyz[IZ];
    r2 += (z - droplet_z) * (z - droplet_z);
  }

  // distance to droplet center
  const auto r = sqrt(r2);

  // liquid hydro state
  auto const & rho_liq = m_initial_states(0)[Hydro::ID];

  // gas hydro state
  auto const & rho_gas = m_initial_states(1)[Hydro::ID];

  auto const & radius = m_staticDropletParams.radius;

  const auto phi_liq = ONE_F / (radius + exp((r - 1) / (Delta * epsilon_coef * dx)));
  const auto phi_gas = ONE_F - phi_liq;

  // const auto delta_p = phi_liq;

  m_Udata(cell_index, m_fm[Hydro::ID0], iOct) =
    phi_liq * m_initial_states(0)[Hydro::ID0] + phi_gas * m_initial_states(1)[Hydro::ID0];
  m_Udata(cell_index, m_fm[Hydro::ID1], iOct) =
    phi_liq * m_initial_states(0)[Hydro::ID1] + phi_gas * m_initial_states(1)[Hydro::ID1];
  m_Udata(cell_index, m_fm[Hydro::IPHI], iOct) =
    phi_liq * m_initial_states(0)[Hydro::IPHI] + phi_gas * m_initial_states(1)[Hydro::IPHI];

  const auto rho =
    m_Udata(cell_index, m_fm[Hydro::ID0], iOct) + m_Udata(cell_index, m_fm[Hydro::ID1], iOct);

  m_Udata(cell_index, m_fm[Hydro::IU], iOct) = rho * ZERO_F;
  m_Udata(cell_index, m_fm[Hydro::IV], iOct) = rho * ZERO_F;
  if constexpr (dim == 3)
  {
    m_Udata(cell_index, m_fm[Hydro::IW], iOct) = rho * ZERO_F;
  }

  // pure states
  auto eint_liq = m_initial_states(0)[Hydro::IE];
  auto eint_gas = m_initial_states(1)[Hydro::IE];

  auto p_liq = m_eos_wrapper.mixture_pressure(rho_liq, eint_liq, ONE_F);
  // auto p_gas = m_eos_wrapper.mixture_pressure(rho_gas, eint_gas, ZERO_F);

  // TODO (once capilarity is implemented): update p_liq with Laplace term
  // p_liq += delta_p_laplace;
  // update eint_liq
  eint_liq = rho_liq * m_eos_wrapper.mixture_specific_eint(p_liq, rho_liq, ONE_F);

  const auto eint_mixed = phi_liq * eint_liq + phi_gas * eint_gas;

  m_Udata(cell_index, m_fm[Hydro::IE], iOct) = eint_mixed;

} // InitStaticDropletDataFunctor::operator ()

template class InitStaticDropletDataFunctor<2, kalypsso::DefaultDevice>;
template class InitStaticDropletDataFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitStaticDropletRefineFunctor<dim, device_t>::apply(DataArrayBlock_t             Udata,
                                                     FieldMap<models::FiveEq>     fm,
                                                     orchard_key_view_t<device_t> orchard_keys,
                                                     amrflags_view_t              amrflags,
                                                     int32_t                      local_num_octants,
                                                     int                          level_refine,
                                                     ConfigMap const &            config_map)
{
  // iterate functor for refinement
  InitStaticDropletRefineFunctor functor(
    Udata, fm, orchard_keys, amrflags, local_num_octants, level_refine, config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_five_eq::InitStaticDropletRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_five_eq::InitStaticDropletRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

} // InitStaticDropletRefineFunctor::apply

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitStaticDropletRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                          const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
} // InitStaticDropletRefineFunctor::operator ()

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitStaticDropletRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
                                                          const size_t & iOct) const
{

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // get block level
  const auto level = orchard_key_t<dim>::level(key);

  // compute block length (in real space units)
  const auto block_length = compute_block_length<dim>(level) * m_scaling_factor;

  // default : do nothing, i.e. neither refine or coarsen
  auto flag = AMRContextBase::KALYPSSO_DO_NOTHING;

  // only look at level - 1
  if (level == m_level_refine)
  {

    // compute physical x,y,z for the block center
    constexpr auto centering = true;
    const auto     xyz_vertex = orchard_key_to_vertex_coord<dim>(key, centering);
    const auto     xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

    // compute distance to droplet surface
    auto const & droplet_x = m_staticDropletParams.x;
    auto const & droplet_y = m_staticDropletParams.y;
    auto         d2 =
      (xyz[IX] - droplet_x) * (xyz[IX] - droplet_x) + (xyz[IY] - droplet_y) * (xyz[IY] - droplet_y);

    if constexpr (dim == 3)
    {
      auto const & droplet_z = m_staticDropletParams.z;
      d2 += (xyz[IZ] - droplet_z) * (xyz[IZ] - droplet_z);
    }

    auto const & droplet_radius = m_staticDropletParams.radius;
    if (fabs(sqrt(d2) - droplet_radius) < (block_length * KALYPSSO_NUM(1.25)))
    {
      flag = AMRContextBase::KALYPSSO_DO_REFINE;
    }

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // InitStaticDropletRefineFunctor::operator ()

template class InitStaticDropletRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitStaticDropletRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitStaticDroplet<dim, device_t>::apply(SolverGodunovFiveEq<dim, device_t> & solver)
{

  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const int         level_min = solver.hydro_params().level_min;
  const int         level_max = solver.hydro_params().level_max;

  const auto nb_regions = InitStaticDropletDataFunctor<dim, device_t>::NB_REGIONS;
  auto       initial_states = get_initial_states<dim, device_t>(config_map, nb_regions);

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitStaticDropletDataFunctor<dim, device_t>::apply(solver.U(),
                                                     solver.model().get_fieldmap(),
                                                     solver.mesh_map()->orchard_keys(),
                                                     solver.amr_mesh()->local_num_quadrants(),
                                                     initial_states,
                                                     config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::SAME_AS_REGULAR_DYNAMICS)
  {
    // iterate several refinements
    int level = level_min;
    while (level < level_max)
    {

      //
      // 1. apply amr cycle using regular refine criterion
      //
      solver.do_amr_cycle();

      //
      // 2. update Udata
      //
      InitStaticDropletDataFunctor<dim, device_t>::apply(solver.U(),
                                                         solver.model().get_fieldmap(),
                                                         solver.mesh_map()->orchard_keys(),
                                                         solver.amr_mesh()->local_num_quadrants(),
                                                         initial_states,
                                                         config_map);

      // update level
      ++level;

    } // end while level<level_max
  }
  else
  {
    // iterate several refinements
    int level = level_min;
    while (level < level_max)
    {
      //
      // 1. create context data for AMR cycle
      //
      AMRContext<dim, device_t> amr_context(amr_mesh->forest()->local_num_quadrants);
      auto                      flags_d = amr_context.m_amrflags_d;
      auto                      flags_h = amr_context.m_amrflags_h;

      //
      // 2. compute refine/coarsen flags
      //
      InitStaticDropletRefineFunctor<dim, device_t>::apply(solver.U(),
                                                           solver.model().get_fieldmap(),
                                                           solver.mesh_map()->orchard_keys(),
                                                           flags_d,
                                                           solver.amr_mesh()->local_num_quadrants(),
                                                           level,
                                                           solver.config_map());
      // amr context will adapt mesh on CPU, so we need flags on host up to date
      Kokkos::deep_copy(flags_h, flags_d);

      //
      // 3. apply AMR cycle on device : refine + coarsen + 2:1 balance
      //
      {
        Kokkos::Profiling::ScopedRegion prof("AMR_refinement_device");
        [[maybe_unused]] auto changed = amr_context.adapt_mesh(solver.amr_mesh()->forest());
        KALYPSSO_INFO_ALL("Mesh changed ? {}", static_cast<int>(changed));
      }

      //
      // 4. re-compute update orchard keys
      //
      solver.update_mesh(do_reset_ghosts);

      // 5. resize Udata
      // now we know the size of the mesh, we can allocate memory for
      // heavy data (U, U2, Uhost, ...)
      solver.resize_solver_data();

      //
      // 6. update Udata
      //
      InitStaticDropletDataFunctor<dim, device_t>::apply(solver.U(),
                                                         solver.model().get_fieldmap(),
                                                         solver.mesh_map()->orchard_keys(),
                                                         solver.amr_mesh()->local_num_quadrants(),
                                                         initial_states,
                                                         config_map);

      // update level
      ++level;

    } // end while level<level_max
  }

#ifdef KALYPSSO_CORE_USE_MPI
  // load balancing (= repartitioning) the octree mesh + userdata over the MPI processes.
  // U and U2 will be resized
  solver.do_load_balancing();
#endif

} // InitStaticDroplet::apply

template class InitStaticDroplet<2, kalypsso::DefaultDevice>;
template class InitStaticDroplet<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso
