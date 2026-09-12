// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitFourQuadrant.cpp
 */

#include <godunov_five_eq/init/InitFourQuadrant.h>
#include <godunov_five_eq/SolverGodunovFiveEq.h>

#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/problems/initRiemannConfig2d.h>
#include <kalypsso/core/multimaterial_utils.h>
#include <kalypsso/core/eos/MieGruneisenEosArray.h>
#include <kalypsso/core/models/Hydro.h>

namespace kalypsso
{
namespace godunov_five_eq
{

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitFourQuadrantDataFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  orchard_key_view_t<device_t> const & orchard_keys,
  int32_t                              local_num_octants,
  InitialStates<dim, device_t> const & initial_states,
  ConfigMap const &                    config_map)
{
  // data init functor
  InitFourQuadrantDataFunctor functor(
    Udata, orchard_keys, local_num_octants, initial_states, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_five_eq::InitFourQuadrantDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitFourQuadrantDataFunctor::apply

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitFourQuadrantDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
{
  const auto & block_sizes = m_Udata.block_size();

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  // makes enum Hydro::VarId available
  using Hydro = models::FiveEq<dim>;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // initialize with invalid value
  const auto region_id = m_params.get_region_id<dim>(xyz);

  m_Udata(cell_index, Hydro::IAD0, iOct) = m_initial_states(region_id)[Hydro::IAD0];
  m_Udata(cell_index, Hydro::IAD1, iOct) = m_initial_states(region_id)[Hydro::IAD1];
  m_Udata(cell_index, Hydro::ID, iOct) =
    m_Udata(cell_index, Hydro::IAD0, iOct) + m_Udata(cell_index, Hydro::IAD1, iOct);

  m_Udata(cell_index, Hydro::IA0, iOct) = m_initial_states(region_id)[Hydro::IA0];
  m_Udata(cell_index, Hydro::IA1, iOct) = m_initial_states(region_id)[Hydro::IA1];
  m_Udata(cell_index, Hydro::IU, iOct) = m_initial_states(region_id)[Hydro::IU];
  m_Udata(cell_index, Hydro::IV, iOct) = m_initial_states(region_id)[Hydro::IV];
  if constexpr (dim == 3)
    m_Udata(cell_index, Hydro::IW, iOct) = m_initial_states(region_id)[Hydro::IW];
  m_Udata(cell_index, Hydro::IE, iOct) = m_initial_states(region_id)[Hydro::IE];

} // InitFourQuadrantDataFunctor::operator ()

template class InitFourQuadrantDataFunctor<2, kalypsso::DefaultDevice>;
template class InitFourQuadrantDataFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitFourQuadrantRefineFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  orchard_key_view_t<device_t> const & orchard_keys,
  amrflags_view_t const &              amrflags,
  int32_t                              local_num_octants,
  int                                  level_refine,
  ConfigMap const &                    config_map)
{
  // iterate functor for refinement
  InitFourQuadrantRefineFunctor functor(
    Udata, orchard_keys, amrflags, local_num_octants, level_refine, config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_five_eq::InitFourQuadrantRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_five_eq::InitFourQuadrantRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

} // InitFourQuadrantRefineFunctor::apply

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitFourQuadrantRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                         const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
} // InitFourQuadrantRefineFunctor::operator ()

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitFourQuadrantRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
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

    // compute distance to interface
    const auto dx = fabs(xyz[IX] - m_params.pos[IX]);
    const auto dy = fabs(xyz[IY] - m_params.pos[IY]);

    if (dx < (block_length * KALYPSSO_NUM(0.95)) or dy < (block_length * KALYPSSO_NUM(0.95)))
      flag = AMRContextBase::KALYPSSO_DO_REFINE;

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // InitFourQuadrantRefineFunctor::operator ()

template class InitFourQuadrantRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitFourQuadrantRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitFourQuadrant<dim, device_t>::apply(SolverGodunovFiveEq<dim, device_t> & solver)
{

  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const int         level_min = solver.hydro_params().level_min;
  const int         level_max = solver.hydro_params().level_max;

  const auto nb_regions = InitFourQuadrantDataFunctor<dim, device_t>::NB_REGIONS;
  const auto four_quad_params = core::FourQuadrantParams(config_map);
  auto       primitive_initial_states = getRiemannConfig<dim>(four_quad_params.config_number);

  InitialStates<dim, device_t> initial_states("Initial state", static_cast<uint>(nb_regions));
  const auto                   initial_states_host = Kokkos::create_mirror_view(initial_states);
  const auto                   eos_wrapper = EosWrapper_t<HostDevice>(config_map);

  for (int i_region = 0; i_region < nb_regions; i_region++)
  {
    using FiveEq = models::FiveEq<dim>;
    using Hydro = core::models::Hydro;

    auto hydro_state = primitive_initial_states[i_region];
    auto i_mat = static_cast<size_t>(get_material_id_in_region(i_region, config_map));

    // primitive variables for FiveEq model
    HydroState<dim> fiveeq_state;

    // copy hydro vars
    fiveeq_state[FiveEq::IU] = hydro_state[Hydro::IU];
    fiveeq_state[FiveEq::IV] = hydro_state[Hydro::IV];
    if constexpr (dim == 3)
      fiveeq_state[FiveEq::IW] = hydro_state[Hydro::IW];

    // setup volume fractions
    const auto section = "region" + std::to_string(i_region);
    const auto alpha0 = config_map.getReal(section, "alpha0", KALYPSSO_NUM(1.0));

    fiveeq_state[FiveEq::IA0] = alpha0;
    fiveeq_state[FiveEq::IA1] = ONE_F - alpha0;

    // setup partial densities, total density and pressure
    fiveeq_state[FiveEq::IAD0] = fiveeq_state[FiveEq::IA0] * hydro_state[Hydro::ID];
    fiveeq_state[FiveEq::IAD1] = fiveeq_state[FiveEq::IA1] * hydro_state[Hydro::ID];
    fiveeq_state[FiveEq::ID] = fiveeq_state[FiveEq::IAD0] + fiveeq_state[FiveEq::IAD1];

    auto const & p = hydro_state[Hydro::IP];
    auto const & rho = fiveeq_state[FiveEq::ID];
    auto const & alpha_rho0 = fiveeq_state[FiveEq::IAD0];
    auto const & alpha_rho1 = fiveeq_state[FiveEq::IAD1];

    // compute specific total energy from pressure and kinetic energy
    const auto eint_specific =
      eos_wrapper.mixture_specific_eint(rho, p, alpha0, 1 - alpha0, alpha_rho0, alpha_rho1);

    auto ekin_specific = HALF_F * (hydro_state[Hydro::IU] * hydro_state[Hydro::IU] +
                                   hydro_state[Hydro::IV] * hydro_state[Hydro::IV]);
    if constexpr (dim == 3)
      ekin_specific += HALF_F * hydro_state[Hydro::IW] * hydro_state[Hydro::IW];

    fiveeq_state[FiveEq::IE] = (eint_specific + ekin_specific) * rho;
    fiveeq_state[FiveEq::IU] = hydro_state[Hydro::IU] * rho;
    fiveeq_state[FiveEq::IV] = hydro_state[Hydro::IV] * rho;
    if constexpr (dim == 3)
      fiveeq_state[FiveEq::IW] = hydro_state[Hydro::IW] * rho;

    // specific variables states
    initial_states_host(i_region) = fiveeq_state;
  }

  Kokkos::deep_copy(initial_states, initial_states_host);

  //////auto       initial_states = get_initial_states<dim, device_t>(config_map, nb_regions);

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitFourQuadrantDataFunctor<dim, device_t>::apply(solver.U(),
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
      InitFourQuadrantDataFunctor<dim, device_t>::apply(solver.U(),
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
      InitFourQuadrantRefineFunctor<dim, device_t>::apply(solver.U(),
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
      InitFourQuadrantDataFunctor<dim, device_t>::apply(solver.U(),
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

} // InitFourQuadrant::apply

template class InitFourQuadrant<2, kalypsso::DefaultDevice>;
template class InitFourQuadrant<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso
