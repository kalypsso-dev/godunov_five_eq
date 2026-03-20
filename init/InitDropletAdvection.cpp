// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitDropletAdvection.cpp
 */

#include <godunov_five_eq/init/InitDropletAdvection.h>
#include <godunov_five_eq/SolverGodunovFiveEq.h>

#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/vof/interface_tracking_utils.h>
#include <kalypsso/core/geometry_utils.h>

namespace kalypsso
{
namespace godunov_five_eq
{

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitDropletAdvectionDataFunctor<dim, device_t>::apply(
  DataArrayBlock_t                     Udata,
  FieldMap<models::FiveEq>             fm,
  orchard_key_view_t<device_t>         orchard_keys,
  int32_t                              local_num_octants,
  InitialStates<dim, device_t> const & initial_states,
  ConfigMap const &                    config_map)
{
  // data init functor
  InitDropletAdvectionDataFunctor functor(
    Udata, fm, orchard_keys, local_num_octants, initial_states, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_five_eq::InitDropletAdvectionDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitDropletAdvectionDataFunctor::apply

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitDropletAdvectionDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
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

  auto const & droplet_x = m_droplet_advection_params.x;
  auto const & droplet_y = m_droplet_advection_params.y;
  auto const & droplet_z = m_droplet_advection_params.z;
  auto const & radius = m_droplet_advection_params.radius;
  auto const   radius2 = radius * radius;

  Kokkos::Array<real_t, dim> bubble_center;
  bubble_center[IX] = droplet_x;
  bubble_center[IY] = droplet_y;
  if constexpr (dim == 3)
  {
    bubble_center[IZ] = droplet_z;
  }

  // determine if current is completely inside or outside of the droplet
  bool is_inside = true;
  bool is_outside = true;
  for (uint8_t i_corner = 0; i_corner < Corner::num_corners<dim>(); i_corner++)
  {
    const auto xyz_vertex_corner =
      orchard_key_to_corner_coord<dim>(key, iCoord, block_sizes[IX], i_corner);
    auto xyz_corner =
      vertex_coord_to_real_space<dim>(xyz_vertex_corner, m_scaling_factor, m_xyz_min);

    // compute distance to droplet center
    auto r2 = (droplet_x - xyz_corner[IX]) * (droplet_x - xyz_corner[IX]) +
              (droplet_y - xyz_corner[IY]) * (droplet_y - xyz_corner[IY]);
    if constexpr (dim == 3)
      r2 += (droplet_z - xyz_corner[IZ]) * (droplet_z - xyz_corner[IZ]);

    if (r2 < radius2)
      is_outside = is_outside & false;
    if (r2 >= radius2)
      is_inside = is_inside & false;
  }

  if (is_inside)
  {
    m_Udata(cell_index, m_fm[Hydro::IPHI], iOct) = m_initial_states(0)[Hydro::IPHI];
    m_Udata(cell_index, m_fm[Hydro::ID0], iOct) = m_initial_states(0)[Hydro::ID0];
    m_Udata(cell_index, m_fm[Hydro::ID1], iOct) = m_initial_states(0)[Hydro::ID1];

    m_Udata(cell_index, m_fm[Hydro::IU], iOct) = m_initial_states(0)[Hydro::IU];
    m_Udata(cell_index, m_fm[Hydro::IV], iOct) = m_initial_states(0)[Hydro::IV];
    if constexpr (dim == 3)
    {
      m_Udata(cell_index, m_fm[Hydro::IW], iOct) = m_initial_states(0)[Hydro::IW];
    }
    m_Udata(cell_index, m_fm[Hydro::IE], iOct) = m_initial_states(0)[Hydro::IE];
  }
  else if (is_outside)
  {
    m_Udata(cell_index, m_fm[Hydro::IPHI], iOct) = m_initial_states(1)[Hydro::IPHI];
    m_Udata(cell_index, m_fm[Hydro::ID0], iOct) = m_initial_states(1)[Hydro::ID0];
    m_Udata(cell_index, m_fm[Hydro::ID1], iOct) = m_initial_states(1)[Hydro::ID1];

    m_Udata(cell_index, m_fm[Hydro::IU], iOct) = m_initial_states(1)[Hydro::IU];
    m_Udata(cell_index, m_fm[Hydro::IV], iOct) = m_initial_states(1)[Hydro::IV];
    if constexpr (dim == 3)
    {
      m_Udata(cell_index, m_fm[Hydro::IW], iOct) = m_initial_states(1)[Hydro::IW];
    }
    m_Udata(cell_index, m_fm[Hydro::IE], iOct) = m_initial_states(1)[Hydro::IE];
  }
  else
  {
    // current cell intersect the droplet border
    Kokkos::Array<real_t, dim> normal;
    real_t                     alpha;
    get_tangent_to_sphere(bubble_center, radius, xyz, normal, alpha);

    // compute volume fraction of droplet
    const auto vf =
      vof::compute_volume_fraction_of_rect_below_plane(normal, alpha, xyz - dx / 2, xyz + dx / 2);

    m_Udata(cell_index, m_fm[Hydro::IPHI], iOct) =
      vf * m_initial_states(0)[Hydro::IPHI] + (ONE_F - vf) * m_initial_states(1)[Hydro::IPHI];
    m_Udata(cell_index, m_fm[Hydro::ID0], iOct) = vf * m_initial_states(0)[Hydro::ID];
    m_Udata(cell_index, m_fm[Hydro::ID1], iOct) = (ONE_F - vf) * m_initial_states(1)[Hydro::ID];

    m_Udata(cell_index, m_fm[Hydro::IU], iOct) =
      vf * m_initial_states(0)[Hydro::IU] + (ONE_F - vf) * m_initial_states(1)[Hydro::IU];
    m_Udata(cell_index, m_fm[Hydro::IV], iOct) =
      vf * m_initial_states(0)[Hydro::IV] + (ONE_F - vf) * m_initial_states(1)[Hydro::IV];

    if constexpr (dim == 3)
    {
      m_Udata(cell_index, m_fm[Hydro::IW], iOct) =
        vf * m_initial_states(0)[Hydro::IW] + (ONE_F - vf) * m_initial_states(1)[Hydro::IW];
    }

    real_t ekin[2]{ ZERO_F, ZERO_F };
    real_t eint[2]{ ZERO_F, ZERO_F };
    for (int i = 0; i < 2; ++i)
    {
      ekin[i] = m_initial_states(i)[Hydro::IU] * m_initial_states(i)[Hydro::IU] +
                m_initial_states(i)[Hydro::IV] * m_initial_states(i)[Hydro::IV];
      if constexpr (dim == 3)
      {
        ekin[i] += m_initial_states(i)[Hydro::IW] * m_initial_states(i)[Hydro::IW];
      }
      ekin[i] /= (TWO_F * m_initial_states(i)[Hydro::ID]);

      eint[i] = m_initial_states(i)[Hydro::IE] - ekin[i];
    }

    real_t rho_mixed =
      m_Udata(cell_index, m_fm[Hydro::ID0], iOct) + m_Udata(cell_index, m_fm[Hydro::ID1], iOct);
    auto ekin_mixed =
      m_Udata(cell_index, m_fm[Hydro::IU], iOct) * m_Udata(cell_index, m_fm[Hydro::IU], iOct) +
      m_Udata(cell_index, m_fm[Hydro::IV], iOct) * m_Udata(cell_index, m_fm[Hydro::IV], iOct);
    if constexpr (dim == 3)
      ekin_mixed +=
        m_Udata(cell_index, m_fm[Hydro::IW], iOct) * m_Udata(cell_index, m_fm[Hydro::IW], iOct);
    ekin_mixed /= (TWO_F * rho_mixed);

    const auto eint_mixed = vf * eint[0] + (ONE_F - vf) * eint[1];

    m_Udata(cell_index, m_fm[Hydro::IE], iOct) = eint_mixed + ekin_mixed;
  }

} // InitDropletAdvectionDataFunctor::operator ()

template class InitDropletAdvectionDataFunctor<2, kalypsso::DefaultDevice>;
template class InitDropletAdvectionDataFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitDropletAdvectionRefineFunctor<dim, device_t>::apply(DataArrayBlock_t             Udata,
                                                        FieldMap<models::FiveEq>     fm,
                                                        orchard_key_view_t<device_t> orchard_keys,
                                                        amrflags_view_t              amrflags,
                                                        int32_t           local_num_octants,
                                                        int               level_refine,
                                                        ConfigMap const & config_map)
{
  // iterate functor for refinement
  InitDropletAdvectionRefineFunctor functor(
    Udata, fm, orchard_keys, amrflags, local_num_octants, level_refine, config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_five_eq::InitDropletAdvectionRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_five_eq::InitDropletAdvectionRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

} // InitDropletAdvectionRefineFunctor::apply

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitDropletAdvectionRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                             const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
} // InitDropletAdvectionRefineFunctor::operator ()

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitDropletAdvectionRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
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
    auto const & droplet_x = m_droplet_advection_params.x;
    auto const & droplet_y = m_droplet_advection_params.y;
    auto         d2 =
      (xyz[IX] - droplet_x) * (xyz[IX] - droplet_x) + (xyz[IY] - droplet_y) * (xyz[IY] - droplet_y);

    if constexpr (dim == 3)
    {
      auto const & droplet_z = m_droplet_advection_params.z;
      d2 += (xyz[IZ] - droplet_z) * (xyz[IZ] - droplet_z);
    }

    auto const & droplet_radius = m_droplet_advection_params.radius;
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

} // InitDropletAdvectionRefineFunctor::operator ()

template class InitDropletAdvectionRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitDropletAdvectionRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitDropletAdvection<dim, device_t>::apply(SolverGodunovFiveEq<dim, device_t> & solver)
{

  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const int         level_min = solver.hydro_params().level_min;
  const int         level_max = solver.hydro_params().level_max;

  const auto nb_regions = InitDropletAdvectionDataFunctor<dim, device_t>::NB_REGIONS;
  auto       initial_states = get_initial_states<dim, device_t>(config_map, nb_regions);

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitDropletAdvectionDataFunctor<dim, device_t>::apply(solver.U(),
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
      InitDropletAdvectionDataFunctor<dim, device_t>::apply(
        solver.U(),
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
      InitDropletAdvectionRefineFunctor<dim, device_t>::apply(
        solver.U(),
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
      InitDropletAdvectionDataFunctor<dim, device_t>::apply(
        solver.U(),
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

} // InitDropletAdvection::apply

template class InitDropletAdvection<2, kalypsso::DefaultDevice>;
template class InitDropletAdvection<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso
