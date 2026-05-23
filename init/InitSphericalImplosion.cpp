// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitSphericalImplosion.cpp
 */

#include <godunov_five_eq/init/InitSphericalImplosion.h>
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
InitSphericalImplosionDataFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  orchard_key_view_t<device_t> const & orchard_keys,
  int32_t                              local_num_octants,
  InitialStates<dim, device_t> const & initial_states,
  ConfigMap const &                    config_map)
{
  // data init functor
  InitSphericalImplosionDataFunctor functor(
    Udata, orchard_keys, local_num_octants, initial_states, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_five_eq::InitSphericalImplosionDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitSphericalImplosionDataFunctor::apply

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION HydroState<dim>
                       InitSphericalImplosionDataFunctor<dim, device_t>::update_initial_state1(
  Kokkos::Array<real_t, dim> const & normal,
  real_t                             v_theta) const
{
  // makes enum Hydro::VarId available
  using Hydro = models::FiveEq<dim>;

  HydroState<dim> initial_state1;

  initial_state1[Hydro::IAD0] = m_initial_states(1)[Hydro::IAD0];
  initial_state1[Hydro::IAD1] = m_initial_states(1)[Hydro::IAD1];
  initial_state1[Hydro::ID] = initial_state1[Hydro::IAD0] + initial_state1[Hydro::IAD1];
  initial_state1[Hydro::IA0] = m_initial_states(1)[Hydro::IA0];
  initial_state1[Hydro::IA1] = m_initial_states(1)[Hydro::IA1];

  initial_state1[Hydro::IU] = m_initial_states(1)[Hydro::IU] + normal[IX] * v_theta;
  initial_state1[Hydro::IV] = m_initial_states(1)[Hydro::IV] + normal[IY] * v_theta;
  if constexpr (dim == 3)
  {
    initial_state1[Hydro::IW] = m_initial_states(1)[Hydro::IW] + normal[IZ] * v_theta;
  }

  auto ekin_old = m_initial_states(1)[Hydro::IU] * m_initial_states(1)[Hydro::IU];
  ekin_old += m_initial_states(1)[Hydro::IV] * m_initial_states(1)[Hydro::IV];
  if constexpr (dim == 3)
  {
    ekin_old += m_initial_states(1)[Hydro::IW] * m_initial_states(1)[Hydro::IW];
  }
  ekin_old = HALF_F * ekin_old / initial_state1[Hydro::ID];

  auto ekin_new = initial_state1[Hydro::IU] * initial_state1[Hydro::IU];
  ekin_new += initial_state1[Hydro::IV] * initial_state1[Hydro::IV];
  if constexpr (dim == 3)
  {
    ekin_new += initial_state1[Hydro::IW] * initial_state1[Hydro::IW];
  }
  ekin_new = HALF_F * ekin_new / initial_state1[Hydro::ID];

  initial_state1[Hydro::IE] = m_initial_states(1)[Hydro::IE] - ekin_old + ekin_new;

  return initial_state1;

} // InitSphericalImplosionDataFunctor<dim, device_t>::update_initial_state1

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitSphericalImplosionDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
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
  const auto dx =
    compute_cell_length<dim>(static_cast<uint8_t>(m_level_max), block_sizes[IX]) * m_scaling_factor;

  // clang-format off
  auto const & r_out        = m_spherical_implosion_params.r_out;
  auto const & v_theta      = m_spherical_implosion_params.v_theta;
  const auto   shell_center = m_spherical_implosion_params.shell_center<dim>();
  // clang-format on

  const auto initial_state0 = m_initial_states(0);
  const auto initial_state2 = m_initial_states(2);

  // check if current cell intersect the outer surface
  // the normal vector is always defined as the radial vector
  Kokkos::Array<real_t, dim> normal;
  real_t                     alpha;
  get_tangent_to_sphere(shell_center, r_out, xyz, normal, alpha);

  auto vol_frac1 =
    vof::compute_volume_fraction_of_rect_below_plane(normal, alpha, xyz - dx / 2, xyz + dx / 2);

  // apply radial velocity modification on initial state for region 1
  const auto initial_state1 = update_initial_state1(normal, v_theta);

  // compute initial state as a linear combination of state 0 and 1
  real_t phi0 = ONE_F;
  real_t phi1 = ZERO_F;
  real_t phi2 = ZERO_F;

  if (vol_frac1 > ZERO_F) // We are in or near the shell outer surface
  {
    // we are in mixed cell between region 0 and 1
    phi1 = vol_frac1;
    phi0 = ONE_F - vol_frac1;
  }

  // check if we are near inner interface
  m_spherical_implosion_params.get_interface_tangent(xyz, normal, alpha);

  auto vol_frac2 =
    vof::compute_volume_fraction_of_rect_below_plane(normal, alpha, xyz - dx / 2, xyz + dx / 2);

  if (vol_frac2 > ZERO_F) // We are in or near the shell outer surface
  {
    // we are in mixed cell between region 0 and 1
    phi2 = vol_frac1 * vol_frac2;
    phi1 = vol_frac1 * (ONE_F - vol_frac2);
  }

  // init state is a linear combination of state 0, 1 and 2
  // clang-format off
  const auto initial_state =
    phi0 * initial_state0 +
    phi1 * initial_state1 +
    phi2 * initial_state2;
  // clang-format on

  // Fill with the outer region state
  m_Udata(cell_index, Hydro::IAD0, iOct) = initial_state[Hydro::IAD0];
  m_Udata(cell_index, Hydro::IAD1, iOct) = initial_state[Hydro::IAD1];
  m_Udata(cell_index, Hydro::ID, iOct) =
    m_Udata(cell_index, Hydro::IAD0, iOct) + m_Udata(cell_index, Hydro::IAD1, iOct);
  m_Udata(cell_index, Hydro::IA0, iOct) = initial_state[Hydro::IA0];
  m_Udata(cell_index, Hydro::IA1, iOct) = initial_state[Hydro::IA1];
  m_Udata(cell_index, Hydro::IU, iOct) = initial_state[Hydro::IU];
  m_Udata(cell_index, Hydro::IV, iOct) = initial_state[Hydro::IV];
  if constexpr (dim == 3)
  {
    m_Udata(cell_index, Hydro::IW, iOct) = initial_state[Hydro::IW];
  }

  const auto ekin0 = ekin_from_conservative_var_state<dim>(initial_state0);
  const auto ekin1 = ekin_from_conservative_var_state<dim>(initial_state1);
  const auto ekin2 = ekin_from_conservative_var_state<dim>(initial_state2);

  const auto ekin_mixed = ekin_from_conservative_var_state<dim>(initial_state);

  const auto eint_mixed = phi0 * (initial_state0[Hydro::IE] - ekin0) +
                          phi1 * (initial_state1[Hydro::IE] - ekin1) +
                          phi2 * (initial_state2[Hydro::IE] - ekin2);

  m_Udata(cell_index, Hydro::IE, iOct) = eint_mixed + ekin_mixed;

} // InitSphericalImplosionDataFunctor::operator ()

template class InitSphericalImplosionDataFunctor<2, kalypsso::DefaultDevice>;
template class InitSphericalImplosionDataFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitSphericalImplosionRefineFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  orchard_key_view_t<device_t> const & orchard_keys,
  amrflags_view_t const &              amrflags,
  int32_t                              local_num_octants,
  int                                  level_refine,
  ConfigMap const &                    config_map)
{
  // iterate functor for refinement
  InitSphericalImplosionRefineFunctor functor(
    Udata, orchard_keys, amrflags, local_num_octants, level_refine, config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_five_eq::InitSphericalImplosionRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_five_eq::InitSphericalImplosionRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

} // InitSphericalImplosionRefineFunctor::apply

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitSphericalImplosionRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                               const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
} // InitSphericalImplosionRefineFunctor::operator ()

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitSphericalImplosionRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
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

    // get shell center
    const auto shell_center = m_spherical_implosion_params.shell_center<dim>();

    // compute distance to shell surface
    auto d2 = (xyz[IX] - shell_center[IX]) * (xyz[IX] - shell_center[IX]) +
              (xyz[IY] - shell_center[IY]) * (xyz[IY] - shell_center[IY]);

    if constexpr (dim == 3)
    {
      d2 += (xyz[IZ] - shell_center[IZ]) * (xyz[IZ] - shell_center[IZ]);
    }

    auto const & r_in = m_spherical_implosion_params.r_in;
    auto const & r_out = m_spherical_implosion_params.r_out;
    if (fabs(sqrt(d2) - r_in) < (block_length * KALYPSSO_NUM(1.25)) or
        fabs(sqrt(d2) - r_out) < (block_length * KALYPSSO_NUM(1.25)))
    {
      flag = AMRContextBase::KALYPSSO_DO_REFINE;
    }

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // InitSphericalImplosionRefineFunctor::operator ()

template class InitSphericalImplosionRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitSphericalImplosionRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
template <size_t dim, typename device_t>
void
InitSphericalImplosion<dim, device_t>::apply(SolverGodunovFiveEq<dim, device_t> & solver)
{

  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const int         level_min = solver.hydro_params().level_min;
  const int         level_max = solver.hydro_params().level_max;

  const auto nb_regions = InitSphericalImplosionDataFunctor<dim, device_t>::NB_REGIONS;
  auto       initial_states = get_initial_states<dim, device_t>(config_map, nb_regions);

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitSphericalImplosionDataFunctor<dim, device_t>::apply(solver.U(),
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
      InitSphericalImplosionDataFunctor<dim, device_t>::apply(
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
      InitSphericalImplosionRefineFunctor<dim, device_t>::apply(
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
      InitSphericalImplosionDataFunctor<dim, device_t>::apply(
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

} // InitSphericalImplosion::apply

template class InitSphericalImplosion<2, kalypsso::DefaultDevice>;
template class InitSphericalImplosion<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso
