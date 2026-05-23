// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitRichtmyerMeshkov.cpp
 */

#include <godunov_five_eq/init/InitRichtmyerMeshkov.h>
#include <godunov_five_eq/SolverGodunovFiveEq.h>

#include <kalypsso/core/vof/interface_tracking_utils.h>
#include <kalypsso/core/region_utils.h>

namespace kalypsso
{
namespace godunov_five_eq
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitRichtmyerMeshkovDataFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  orchard_key_view_t<device_t> const & orchard_keys,
  int32_t                              local_num_octants,
  InitialStates<dim, device_t> const & initial_states,
  ConfigMap const &                    config_map)
{
  // data init functor
  InitRichtmyerMeshkovDataFunctor functor(
    Udata, orchard_keys, local_num_octants, initial_states, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_five_eq::InitRichtmyerMeshkovDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitRichtmyerMeshkovDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitRichtmyerMeshkovDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  // makes enum Hydro::VarId available
  using Hydro = models::FiveEq<dim>;

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
  const auto level = orchard_key_t<dim>::level(key);
  const auto dx = compute_cell_length<dim>(level, block_sizes[IX]) * m_scaling_factor;


  const auto region_id = point_to_region(xyz);

  // in the following we strongly assume that amr level is sufficiently high so that
  // a given cell can't belong to all three regions at the same time

  // array of region id (one per corner)
  Kokkos::Array<int, Corner::num_corners<dim>()> corner_regions;
  compute_corner_to_region(iCoord, key, corner_regions);

  bool init_as_pure_cell = is_cell_fully_inside_region<dim>(corner_regions) or
                           (does_cell_overlap_with_region<dim>(corner_regions, 0) and
                            does_cell_overlap_with_region<dim>(corner_regions, 1));

  if (init_as_pure_cell)
  {
    m_Udata(cell_index, Hydro::IA0, iOct) = m_initial_states(region_id)[Hydro::IA0];
    m_Udata(cell_index, Hydro::IA1, iOct) = m_initial_states(region_id)[Hydro::IA1];
    m_Udata(cell_index, Hydro::IAD0, iOct) = m_initial_states(region_id)[Hydro::IAD0];
    m_Udata(cell_index, Hydro::IAD1, iOct) = m_initial_states(region_id)[Hydro::IAD1];
    m_Udata(cell_index, Hydro::ID, iOct) =
      m_Udata(cell_index, Hydro::IAD0, iOct) + m_Udata(cell_index, Hydro::IAD1, iOct);

    m_Udata(cell_index, Hydro::IU, iOct) = m_initial_states(region_id)[Hydro::IU];
    m_Udata(cell_index, Hydro::IV, iOct) = m_initial_states(region_id)[Hydro::IV];
    if constexpr (dim == 3)
      m_Udata(cell_index, Hydro::IW, iOct) = m_initial_states(region_id)[Hydro::IW];
    m_Udata(cell_index, Hydro::IE, iOct) = m_initial_states(region_id)[Hydro::IE];
  }
  else
  {
    // initial mixed cells at interface between regions 1 and 2 (i.e. the sine wave interface)

    // first get sine wave tangent at the same y as cell center
    Kokkos::Array<real_t, dim> normal;
    real_t                     alpha;
    m_rm_params.get_interface_tangent(xyz, normal, alpha);

    // compute volume fraction of region 1 (filled with material 0)
    const auto vf =
      vof::compute_volume_fraction_of_rect_below_plane(normal, alpha, xyz - dx / 2, xyz + dx / 2);

    m_Udata(cell_index, Hydro::IA0, iOct) =
      vf * m_initial_states(1)[Hydro::IA0] + (ONE_F - vf) * m_initial_states(2)[Hydro::IA0];
    m_Udata(cell_index, Hydro::IA1, iOct) =
      vf * m_initial_states(1)[Hydro::IA1] + (ONE_F - vf) * m_initial_states(2)[Hydro::IA1];

    m_Udata(cell_index, Hydro::IAD0, iOct) =
      vf * m_initial_states(1)[Hydro::IAD0] + (ONE_F - vf) * m_initial_states(2)[Hydro::IAD0];
    m_Udata(cell_index, Hydro::IAD1, iOct) =
      vf * m_initial_states(1)[Hydro::IAD1] + (ONE_F - vf) * m_initial_states(2)[Hydro::IAD1];
    m_Udata(cell_index, Hydro::ID, iOct) =
      m_Udata(cell_index, Hydro::IAD0, iOct) + m_Udata(cell_index, Hydro::IAD1, iOct);

    m_Udata(cell_index, Hydro::IU, iOct) =
      vf * m_initial_states(1)[Hydro::IU] + (ONE_F - vf) * m_initial_states(2)[Hydro::IU];
    m_Udata(cell_index, Hydro::IV, iOct) =
      vf * m_initial_states(1)[Hydro::IV] + (ONE_F - vf) * m_initial_states(2)[Hydro::IV];

    if constexpr (dim == 3)
    {
      m_Udata(cell_index, Hydro::IW, iOct) =
        vf * m_initial_states(1)[Hydro::IW] + (ONE_F - vf) * m_initial_states(2)[Hydro::IW];
    }

    real_t eint[3]{ ZERO_F, ZERO_F, ZERO_F };
    for (int i = 1; i < 3; ++i)
    {
      auto ekin = m_initial_states(i)[Hydro::IU] * m_initial_states(i)[Hydro::IU] +
                  m_initial_states(i)[Hydro::IV] * m_initial_states(i)[Hydro::IV];
      if constexpr (dim == 3)
      {
        ekin += m_initial_states(i)[Hydro::IW] * m_initial_states(i)[Hydro::IW];
      }
      ekin /= (TWO_F * m_initial_states(i)[Hydro::ID]);

      eint[i] = m_initial_states(i)[Hydro::IE] - ekin;
    }

    real_t rho_mixed = m_Udata(cell_index, Hydro::ID, iOct);
    auto ekin_mixed = m_Udata(cell_index, Hydro::IU, iOct) * m_Udata(cell_index, Hydro::IU, iOct) +
                      m_Udata(cell_index, Hydro::IV, iOct) * m_Udata(cell_index, Hydro::IV, iOct);
    if constexpr (dim == 3)
      ekin_mixed += m_Udata(cell_index, Hydro::IW, iOct) * m_Udata(cell_index, Hydro::IW, iOct);
    ekin_mixed /= (TWO_F * rho_mixed);

    const auto eint_mixed = vf * eint[1] + (ONE_F - vf) * eint[2];

    m_Udata(cell_index, Hydro::IE, iOct) = eint_mixed + ekin_mixed;
  }

} // InitRichtmyerMeshkovDataFunctor::operator ()

template class InitRichtmyerMeshkovDataFunctor<2, kalypsso::DefaultDevice>;
template class InitRichtmyerMeshkovDataFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitRichtmyerMeshkovRefineFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  orchard_key_view_t<device_t> const & orchard_keys,
  amrflags_view_t const &              amrflags,
  int32_t                              local_num_octants,
  int                                  level_refine,
  ConfigMap const &                    config_map)
{
  // iterate functor for refinement
  InitRichtmyerMeshkovRefineFunctor functor(
    Udata, orchard_keys, amrflags, local_num_octants, level_refine, config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_five_eq::InitRichtmyerMeshkovRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_five_eq::InitRichtmyerMeshkovRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

} // InitRichtmyerMeshkovRefineFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitRichtmyerMeshkovRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                             const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
} // InitRichtmyerMeshkovRefineFunctor::operator ()

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitRichtmyerMeshkovRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
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

    // compute distance to shock interface
    const auto d1 = fabs(xyz[IX] - m_rm_params.x_shock);

    // compute distance to material interface
    const auto d2 = fabs(xyz[IX] - m_rm_params.x_material(xyz[IY]));


    if (d1 < (block_length * KALYPSSO_NUM(0.95)) or d2 < (block_length * KALYPSSO_NUM(0.95)))
      flag = AMRContextBase::KALYPSSO_DO_REFINE;

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // InitRichtmyerMeshkovRefineFunctor::operator ()

template class InitRichtmyerMeshkovRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitRichtmyerMeshkovRefineFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitRichtmyerMeshkov<dim, device_t>::apply(SolverGodunovFiveEq<dim, device_t> & solver)
{

  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const int         level_min = solver.hydro_params().level_min;
  const int         level_max = solver.hydro_params().level_max;

  const auto nb_regions = InitRichtmyerMeshkovDataFunctor<dim, device_t>::NB_REGIONS;
  auto       initial_states = get_initial_states<dim, device_t>(config_map, nb_regions);

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitRichtmyerMeshkovDataFunctor<dim, device_t>::apply(solver.U(),
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
      InitRichtmyerMeshkovDataFunctor<dim, device_t>::apply(
        solver.U(),
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
      InitRichtmyerMeshkovRefineFunctor<dim, device_t>::apply(
        solver.U(),
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
      InitRichtmyerMeshkovDataFunctor<dim, device_t>::apply(
        solver.U(),
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

} // InitRichtmyerMeshkov::apply

template class InitRichtmyerMeshkov<2, kalypsso::DefaultDevice>;
template class InitRichtmyerMeshkov<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso
