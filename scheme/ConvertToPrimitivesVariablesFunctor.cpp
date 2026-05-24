// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ConvertToPrimitivesVariablesFunctor.cpp
 */
#include <godunov_five_eq/scheme/ConvertToPrimitivesVariablesFunctor.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
ConvertToPrimitivesVariablesFunctor<dim, device_t>::ConvertToPrimitivesVariablesFunctor(
  StencilHelper_t const &         stencil_helper,
  AMRMeshInfo const &             amr_mesh_info,
  int32_t                         iOct_begin,
  DataArrayBlock_t const &        userdata_in,
  DataArrayGhostedBlock_t const & userdata_out,
  HydroSettings const &           hydro_settings,
  EosWrapper_t<device_t> const &  eos,
  CellCenteredProlongationType    prolongation)
  : m_stencil_helper(stencil_helper)
  , m_mirror_orchard_keys_device()
  , m_amr_mesh_info(amr_mesh_info)
  , m_iOct_begin(iOct_begin)
  , m_userdata_in(userdata_in)
  , m_userdata_out(userdata_out)
  , m_hydro_settings(hydro_settings)
  , m_eos(eos)
  , m_prolongation(prolongation)
  , m_cons_interpol()
{
  KOKKOS_ASSERT(userdata_in.block_size() == userdata_out.block_size() &&
                "userdata_in and userdata_out must have the same block sizes.");
}

// ==============================================================
// ==============================================================
// same as above, but specifying also the mirror keys array
template <size_t dim, typename device_t>
ConvertToPrimitivesVariablesFunctor<dim, device_t>::ConvertToPrimitivesVariablesFunctor(
  StencilHelper_t const &         stencil_helper,
  orchard_key_view_t const &      mirror_orchard_keys,
  AMRMeshInfo const &             amr_mesh_info,
  DataArrayBlock_t const &        userdata_in,
  DataArrayGhostedBlock_t const & userdata_out,
  HydroSettings const &           hydro_settings,
  EosWrapper_t<device_t> const &  eos,
  CellCenteredProlongationType    prolongation)
  : m_stencil_helper(stencil_helper)
  , m_mirror_orchard_keys_device(mirror_orchard_keys)
  , m_amr_mesh_info(amr_mesh_info)
  , m_iOct_begin(0) // not used when processing mirrors quad
  , m_userdata_in(userdata_in)
  , m_userdata_out(userdata_out)
  , m_hydro_settings(hydro_settings)
  , m_eos(eos)
  , m_prolongation(prolongation)
  , m_cons_interpol()
{
  KOKKOS_ASSERT(userdata_in.block_size() == userdata_out.block_size() &&
                "userdata_in and userdata_out must have the same block sizes.");
}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::apply_on_group(
  ConfigMap const &                config_map,
  amr_hashmap_t const &            amr_hashmap,
  orchard_key_view_t const &       orchard_keys,
  AMRMeshInfo const &              amr_mesh_info,
  int32_t                          iOct_begin,
  int32_t                          num_octants_in_group,
  DataArrayBlock_t const &         userdata_in,
  DataArrayGhostedBlock_t const &  userdata_out,
  brick_size_t<dim> const &        brick_sizes,
  Kokkos::Array<bool, dim> const & is_brick_periodic,
  HydroSettings const &            hydro_settings,
  EosWrapper_t<device_t> const &   eos)
{
  // make sure the range of octants to process is valid
  assertm((iOct_begin + num_octants_in_group) <= amr_mesh_info.local_num_quadrants(),
          "Invalid range of octants to process");

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, userdata_in.block_size(), brick_sizes, is_brick_periodic);

  ConvertToPrimitivesVariablesFunctor<dim, device_t> functor(
    stencil_helper,
    amr_mesh_info,
    iOct_begin,
    userdata_in,
    userdata_out,
    hydro_settings,
    eos,
    get_cell_prolongation_type(config_map));

  const auto nbCellsPerGhostedLeaf = userdata_out.num_cells();
  const auto nbCellsTotal = num_octants_in_group * nbCellsPerGhostedLeaf;

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ConvertToPrimitivesVariablesFunctor",
                       Kokkos::RangePolicy<exec_space, TagComputeAllQuad>(0, nbCellsTotal),
                       functor);

} // apply_on_group

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::apply_in_mirrors(
  ConfigMap const &                config_map,
  amr_hashmap_t const &            amr_hashmap,
  orchard_key_view_t const &       orchard_keys,
  orchard_key_view_t const &       mirror_orchard_keys,
  AMRMeshInfo const &              amr_mesh_info,
  DataArrayBlock_t const &         userdata_in,
  DataArrayGhostedBlock_t const &  userdata_out,
  brick_size_t<dim> const &        brick_sizes,
  Kokkos::Array<bool, dim> const & is_brick_periodic,
  HydroSettings const &            hydro_settings,
  EosWrapper_t<device_t> const &   eos)
{

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, userdata_in.block_size(), brick_sizes, is_brick_periodic);

  ConvertToPrimitivesVariablesFunctor<dim, device_t> functor(
    stencil_helper,
    mirror_orchard_keys,
    amr_mesh_info,
    userdata_in,
    userdata_out,
    hydro_settings,
    eos,
    get_cell_prolongation_type(config_map));

  const auto num_mirrors = mirror_orchard_keys.extent(0);
  const auto nbCellsPerGhostedLeaf = userdata_out.num_cells();
  const auto nbCellsTotal = num_mirrors * static_cast<size_t>(nbCellsPerGhostedLeaf);

  assertm(num_mirrors == static_cast<size_t>(amr_mesh_info.local_num_mirrors()),
          "wrong number of mirror quads.");

  // for AMR tree leaf, explore the neighbor block
  Kokkos::parallel_for("ConvertToPrimitivesVariablesFunctor",
                       Kokkos::RangePolicy<exec_space, TagComputeMirrorQuad>(0, nbCellsTotal),
                       functor);

} // apply_in_mirrors

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION HydroState<dim>
ConvertToPrimitivesVariablesFunctor<dim, device_t>::get_conservative_vars(const int32_t cellindex,
                                                                          const iOct_t  iOct) const
{
  HydroState<dim> uLoc; // conservative variables in current cell

  uLoc[Hydro::ID] = m_userdata_in(cellindex, Hydro::ID, iOct);
  uLoc[Hydro::IE] = m_userdata_in(cellindex, Hydro::IE, iOct);
  uLoc[Hydro::IU] = m_userdata_in(cellindex, Hydro::IU, iOct);
  uLoc[Hydro::IV] = m_userdata_in(cellindex, Hydro::IV, iOct);

  if constexpr (dim == 3)
  {
    uLoc[Hydro::IW] = m_userdata_in(cellindex, Hydro::IW, iOct);
  }

  uLoc[Hydro::IA0] = m_userdata_in(cellindex, Hydro::IA0, iOct);
  uLoc[Hydro::IAD0] = m_userdata_in(cellindex, Hydro::IAD0, iOct);

  uLoc[Hydro::IA1] = m_userdata_in(cellindex, Hydro::IA1, iOct);
  uLoc[Hydro::IAD1] = m_userdata_in(cellindex, Hydro::IAD1, iOct);

  return uLoc;

} // get_conservative_vars

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION HydroState<dim>
                       ConvertToPrimitivesVariablesFunctor<dim, device_t>::get_conservative_vars(
  CellLocation_t const & cell_loc) const
{
  const auto   cellindex_in = cell_loc.cellindex(m_userdata_in.block_size());
  const auto & iOct_in = cell_loc.iOct;

  return get_conservative_vars(cellindex_in, iOct_in);

} // get_conservative_vars

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION HydroState<dim>
ConvertToPrimitivesVariablesFunctor<dim, device_t>::get_conservative_vars_restriction(
  CellLocation_t const & cell_loc) const
{

  HydroState<dim> uLoc; // conservative variables in current cell

  auto const & block_size = m_userdata_in.block_size();

  uLoc[Hydro::IE] =
    m_stencil_helper.compute_siblings_average(cell_loc, block_size, Hydro::IE, m_userdata_in);

  uLoc[Hydro::IU] =
    m_stencil_helper.compute_siblings_average(cell_loc, block_size, Hydro::IU, m_userdata_in);

  uLoc[Hydro::IV] =
    m_stencil_helper.compute_siblings_average(cell_loc, block_size, Hydro::IV, m_userdata_in);

  if constexpr (dim == 3)
  {
    uLoc[Hydro::IW] =
      m_stencil_helper.compute_siblings_average(cell_loc, block_size, Hydro::IW, m_userdata_in);
  }

  uLoc[Hydro::IAD0] =
    m_stencil_helper.compute_siblings_average(cell_loc, block_size, Hydro::IAD0, m_userdata_in);

  uLoc[Hydro::IAD1] =
    m_stencil_helper.compute_siblings_average(cell_loc, block_size, Hydro::IAD1, m_userdata_in);

  uLoc[Hydro::IA0] =
    m_stencil_helper.compute_siblings_average(cell_loc, block_size, Hydro::IA0, m_userdata_in);

  uLoc[Hydro::IA1] = ONE_F - uLoc[Hydro::IA0];

  uLoc[Hydro::ID] = uLoc[Hydro::IAD0] + uLoc[Hydro::IAD1];

  return uLoc;

} // get_conservative_vars_restriction

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::set_primitive_vars(
  const int32_t           cellindex_g,
  const iOct_t            iOct_out,
  HydroState<dim> const & q) const
{
  m_userdata_out(cellindex_g, Hydro::ID, iOct_out) = q[Hydro::ID];
  m_userdata_out(cellindex_g, Hydro::IAD0, iOct_out) = q[Hydro::IAD0];
  m_userdata_out(cellindex_g, Hydro::IAD1, iOct_out) = q[Hydro::IAD1];
  m_userdata_out(cellindex_g, Hydro::IA0, iOct_out) = q[Hydro::IA0];
  m_userdata_out(cellindex_g, Hydro::IA1, iOct_out) = ONE_F - q[Hydro::IA0];
  m_userdata_out(cellindex_g, Hydro::IP, iOct_out) = q[Hydro::IP];
  m_userdata_out(cellindex_g, Hydro::IU, iOct_out) = q[Hydro::IU];
  m_userdata_out(cellindex_g, Hydro::IV, iOct_out) = q[Hydro::IV];
  if constexpr (dim == 3)
  {
    m_userdata_out(cellindex_g, Hydro::IW, iOct_out) = q[Hydro::IW];
  }

} // set_primitive_vars

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION real_t
ConvertToPrimitivesVariablesFunctor<dim, device_t>::extrapol_2d(
  CellLocation<dim> const &       cell_loc_neigh,
  CellLocation<dim> const &       cell_loc_left_x,
  CellLocation<dim> const &       cell_loc_right_x,
  CellLocation<dim> const &       cell_loc_left_y,
  CellLocation<dim> const &       cell_loc_right_y,
  Kokkos::Array<int, dim> const & ixyz,
  int32_t const                   ivar) const
{
  const real_t slope_type = 1;

  // compute limited slopes
  auto const dudx = m_stencil_helper.compute_minmod_slopes(
    cell_loc_neigh, cell_loc_right_x, cell_loc_left_x, ivar, m_userdata_in, slope_type);
  auto const dudy = m_stencil_helper.compute_minmod_slopes(
    cell_loc_neigh, cell_loc_right_y, cell_loc_left_y, ivar, m_userdata_in, slope_type);

  // extrapolate conservative variables
  return m_userdata_in(
           cell_loc_neigh.cellindex(m_userdata_in.block_size()), ivar, cell_loc_neigh.iOct) +
         KALYPSSO_NUM(0.25) * static_cast<real_t>(ixyz[IX]) * dudx +
         KALYPSSO_NUM(0.25) * static_cast<real_t>(ixyz[IY]) * dudy;

} // extrapol_2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION real_t
ConvertToPrimitivesVariablesFunctor<dim, device_t>::extrapol_3d(
  CellLocation<dim> const &       cell_loc_neigh,
  CellLocation<dim> const &       cell_loc_left_x,
  CellLocation<dim> const &       cell_loc_right_x,
  CellLocation<dim> const &       cell_loc_left_y,
  CellLocation<dim> const &       cell_loc_right_y,
  CellLocation<dim> const &       cell_loc_left_z,
  CellLocation<dim> const &       cell_loc_right_z,
  Kokkos::Array<int, dim> const & ixyz,
  int32_t const                   ivar) const
{
  const real_t slope_type = 1;

  // compute limited slopes
  auto const dudx = m_stencil_helper.compute_minmod_slopes(
    cell_loc_neigh, cell_loc_right_x, cell_loc_left_x, ivar, m_userdata_in, slope_type);
  auto const dudy = m_stencil_helper.compute_minmod_slopes(
    cell_loc_neigh, cell_loc_right_y, cell_loc_left_y, ivar, m_userdata_in, slope_type);
  auto const dudz = m_stencil_helper.compute_minmod_slopes(
    cell_loc_neigh, cell_loc_right_z, cell_loc_left_z, ivar, m_userdata_in, slope_type);

  // extrapolate conservative variables
  return m_userdata_in(
           cell_loc_neigh.cellindex(m_userdata_in.block_size()), ivar, cell_loc_neigh.iOct) +
         KALYPSSO_NUM(0.25) * static_cast<real_t>(ixyz[IX]) * dudx +
         KALYPSSO_NUM(0.25) * static_cast<real_t>(ixyz[IY]) * dudy +
         KALYPSSO_NUM(0.25) * static_cast<real_t>(ixyz[IZ]) * dudz;

} // extrapol_3d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION real_t
ConvertToPrimitivesVariablesFunctor<dim, device_t>::interpol_order2(
  CellLocation<dim> const &       cell_loc_neigh,
  Kokkos::Array<int, dim> const & ixyz,
  int32_t const                   ivar) const
{

  if constexpr (dim == 2)
  {
    real_t val[3];

    for (int j = -1; j < 2; ++j)
    {
      // interpolate along X
      auto cell_loc_x_m1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ -1, j });
      auto cell_loc_x_0 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ 0, j });
      auto cell_loc_x_p1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ 1, j });

      val[j + 1] = m_stencil_helper.compute_linear_combination(
        cell_loc_x_m1,
        cell_loc_x_0,
        cell_loc_x_p1,
        cell_loc_neigh.level(),
        ivar,
        m_userdata_in,
        ixyz[IX] == 0 ? m_cons_interpol.COEFS2_L : m_cons_interpol.COEFS2_R);
    }

    return m_cons_interpol.order2(val[0], val[1], val[2], ixyz[IY] == 0);
  }
  else if constexpr (dim == 3)
  {
    real_t valx[3][3];

    for (int k = -1; k < 2; ++k)
    {
      for (int j = -1; j < 2; ++j)
      {
        // interpolate along X
        auto cell_loc_x_m1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ -1, j, k });
        auto cell_loc_x_0 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ 0, j, k });
        auto cell_loc_x_p1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ 1, j, k });

        valx[k + 1][j + 1] = m_stencil_helper.compute_linear_combination(
          cell_loc_x_m1,
          cell_loc_x_0,
          cell_loc_x_p1,
          cell_loc_neigh.level(),
          ivar,
          m_userdata_in,
          ixyz[IX] == 0 ? m_cons_interpol.COEFS2_L : m_cons_interpol.COEFS2_R);
      }
    }
    real_t valy[3];

    valy[0] = m_cons_interpol.order2(valx[0][0], valx[0][1], valx[0][2], ixyz[IY] == 0);
    valy[1] = m_cons_interpol.order2(valx[1][0], valx[1][1], valx[1][2], ixyz[IY] == 0);
    valy[2] = m_cons_interpol.order2(valx[2][0], valx[2][1], valx[2][2], ixyz[IY] == 0);

    return m_cons_interpol.order2(valy[0], valy[1], valy[2], ixyz[IZ] == 0);
  }

  return ZERO_F;

} // interpol_order2

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION real_t
ConvertToPrimitivesVariablesFunctor<dim, device_t>::interpol_order4(
  CellLocation<dim> const &       cell_loc_neigh,
  Kokkos::Array<int, dim> const & ixyz,
  int32_t const                   ivar) const
{

  if constexpr (dim == 2)
  {
    real_t val[5];

    for (int j = -2; j < 3; ++j)
    {
      // interpolate along X
      auto cell_loc_x_m2 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ -2, j });
      auto cell_loc_x_m1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ -1, j });
      auto cell_loc_x_0 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ 0, j });
      auto cell_loc_x_p1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ 1, j });
      auto cell_loc_x_p2 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<2>{ 2, j });

      val[j + 2] = m_stencil_helper.compute_linear_combination(
        cell_loc_x_m2,
        cell_loc_x_m1,
        cell_loc_x_0,
        cell_loc_x_p1,
        cell_loc_x_p2,
        cell_loc_neigh.level(),
        ivar,
        m_userdata_in,
        ixyz[IX] == 0 ? m_cons_interpol.COEFS4_L : m_cons_interpol.COEFS4_R);
    }

    return m_cons_interpol.order4(val[0], val[1], val[2], val[3], val[4], ixyz[IY] == 0);
  }
  else if constexpr (dim == 3)
  {
    real_t valx[5][5];

    for (int k = -2; k < 3; ++k)
    {
      for (int j = -2; j < 3; ++j)
      {
        // interpolate along X
        auto cell_loc_x_m2 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ -2, j, k });
        auto cell_loc_x_m1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ -1, j, k });
        auto cell_loc_x_0 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ 0, j, k });
        auto cell_loc_x_p1 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ 1, j, k });
        auto cell_loc_x_p2 = m_stencil_helper.getNeighLoc(cell_loc_neigh, shift_t<3>{ 2, j, k });

        valx[k + 2][j + 2] = m_stencil_helper.compute_linear_combination(
          cell_loc_x_m2,
          cell_loc_x_m1,
          cell_loc_x_0,
          cell_loc_x_p1,
          cell_loc_x_p2,
          cell_loc_neigh.level(),
          ivar,
          m_userdata_in,
          ixyz[IX] == 0 ? m_cons_interpol.COEFS4_L : m_cons_interpol.COEFS4_R);
      }
    }
    real_t valy[5];

    valy[0] = m_cons_interpol.order4(
      valx[0][0], valx[0][1], valx[0][2], valx[0][3], valx[0][4], ixyz[IY] == 0);
    valy[1] = m_cons_interpol.order4(
      valx[1][0], valx[1][1], valx[1][2], valx[1][3], valx[1][4], ixyz[IY] == 0);
    valy[2] = m_cons_interpol.order4(
      valx[2][0], valx[2][1], valx[2][2], valx[2][3], valx[2][4], ixyz[IY] == 0);
    valy[3] = m_cons_interpol.order4(
      valx[3][0], valx[3][1], valx[3][2], valx[3][3], valx[3][4], ixyz[IY] == 0);
    valy[4] = m_cons_interpol.order4(
      valx[4][0], valx[4][1], valx[4][2], valx[4][3], valx[4][4], ixyz[IY] == 0);

    return m_cons_interpol.order4(valy[0], valy[1], valy[2], valy[3], valy[4], ixyz[IZ] == 0);
  }

  return ZERO_F;

} // interpol_order4

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::fill_inner(int32_t cellindex_in,
                                                               int32_t cellindex_out,
                                                               iOct_t  iOct_global,
                                                               iOct_t  iOct_out) const
{
  // read conservative variables
  const auto uLoc = get_conservative_vars(cellindex_in, iOct_global);

  // compute primitive variables in current cell
  HydroState<dim>             qLoc; // primitive    variables in current cell
  [[maybe_unused]] const auto c =
    models::computePrimitives<dim>(uLoc, qLoc, m_hydro_settings, m_eos);

  // write primitive variables
  set_primitive_vars(cellindex_out, iOct_out, qLoc);

} // fill_inner

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::fill_ghost_copy(
  CellLocation_t const & cell_loc_out,
  CellLocation_t const & cell_loc_in,
  index_t const &        cellindex_out,
  iOct_t const &         iOct_out) const
{

  const bool do_restriction = cell_loc_in.level() == (cell_loc_out.level() + 1);

  // read conservative variables
  const auto uLoc = do_restriction ? get_conservative_vars_restriction(cell_loc_in)
                                   : get_conservative_vars(cell_loc_in);

  // compute primitive variables in current cell
  HydroState<dim>             qLoc; // primitive    variables in current cell
  [[maybe_unused]] const auto c =
    models::computePrimitives<dim>(uLoc, qLoc, m_hydro_settings, m_eos);

  // write primitive variables
  set_primitive_vars(cellindex_out, iOct_out, qLoc);

} // fill_ghost_copy

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<2> const &         cell_loc_neigh,
  coord_t<2> const &              coord_in,
  [[maybe_unused]] iOct_t const & iOct_global,
  index_t const &                 cellindex_out,
  iOct_t const &                  iOct_out) const
{
  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+YDIR));

  // determine local position of current cell inside virtual parent cell using integer coordinates
  // in -1, +1
  const auto ixyz = get_local_position2<dim>(coord_in);

  HydroState<dim> uLoc; // conservative variables in current cell

  uLoc[Hydro::IAD0] = extrapol_2d(cell_loc_neigh,
                                  cell_loc_left_x,
                                  cell_loc_right_x,
                                  cell_loc_left_y,
                                  cell_loc_right_y,
                                  ixyz,
                                  Hydro::IAD0);

  uLoc[Hydro::IAD1] = extrapol_2d(cell_loc_neigh,
                                  cell_loc_left_x,
                                  cell_loc_right_x,
                                  cell_loc_left_y,
                                  cell_loc_right_y,
                                  ixyz,
                                  Hydro::IAD1);

  uLoc[Hydro::ID] = uLoc[Hydro::IAD0] + uLoc[Hydro::IAD1];

  uLoc[Hydro::IA0] = extrapol_2d(cell_loc_neigh,
                                 cell_loc_left_x,
                                 cell_loc_right_x,
                                 cell_loc_left_y,
                                 cell_loc_right_y,
                                 ixyz,
                                 Hydro::IA0);

  uLoc[Hydro::IA1] = ONE_F - uLoc[Hydro::IA0];

  uLoc[Hydro::IE] = extrapol_2d(cell_loc_neigh,
                                cell_loc_left_x,
                                cell_loc_right_x,
                                cell_loc_left_y,
                                cell_loc_right_y,
                                ixyz,
                                Hydro::IE);

  uLoc[Hydro::IU] = extrapol_2d(cell_loc_neigh,
                                cell_loc_left_x,
                                cell_loc_right_x,
                                cell_loc_left_y,
                                cell_loc_right_y,
                                ixyz,
                                Hydro::IU);

  uLoc[Hydro::IV] = extrapol_2d(cell_loc_neigh,
                                cell_loc_left_x,
                                cell_loc_right_x,
                                cell_loc_left_y,
                                cell_loc_right_y,
                                ixyz,
                                Hydro::IV);

  // compute primitive variables in current cell
  HydroState<dim>             qLoc; // primitive    variables in current cell
  [[maybe_unused]] const auto c =
    models::computePrimitives<dim>(uLoc, qLoc, m_hydro_settings, m_eos);

  // write primitive variables
  set_primitive_vars(cellindex_out, iOct_out, qLoc);

} // linear_extrapolate_using_limited_slopes - 2d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::linear_extrapolate_using_limited_slopes(
  CellLocation<3> const &         cell_loc_neigh,
  coord_t<3> const &              coord_in,
  [[maybe_unused]] iOct_t const & iOct_global,
  index_t const &                 cellindex_out,
  iOct_t const &                  iOct_out) const
{

  const auto cell_loc_left_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-XDIR));
  const auto cell_loc_right_x =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+XDIR));

  const auto cell_loc_left_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-YDIR));
  const auto cell_loc_right_y =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+YDIR));

  const auto cell_loc_left_z =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(-ZDIR));
  const auto cell_loc_right_z =
    m_stencil_helper.getNeighLoc(cell_loc_neigh, m_stencil_helper.unit_shift(+ZDIR));

  // determine local position of current cell inside virtual parent cell using integer coordinates
  // in -1, +1
  const auto ixyz = get_local_position2<dim>(coord_in);

  HydroState<dim> uLoc; // conservative variables in current cell

  uLoc[Hydro::IAD0] = extrapol_3d(cell_loc_neigh,
                                  cell_loc_left_x,
                                  cell_loc_right_x,
                                  cell_loc_left_y,
                                  cell_loc_right_y,
                                  cell_loc_left_z,
                                  cell_loc_right_z,
                                  ixyz,
                                  Hydro::IAD0);

  uLoc[Hydro::IAD1] = extrapol_3d(cell_loc_neigh,
                                  cell_loc_left_x,
                                  cell_loc_right_x,
                                  cell_loc_left_y,
                                  cell_loc_right_y,
                                  cell_loc_left_z,
                                  cell_loc_right_z,
                                  ixyz,
                                  Hydro::IAD1);

  uLoc[Hydro::ID] = uLoc[Hydro::IAD0] + uLoc[Hydro::IAD1];

  uLoc[Hydro::IA0] = extrapol_3d(cell_loc_neigh,
                                 cell_loc_left_x,
                                 cell_loc_right_x,
                                 cell_loc_left_y,
                                 cell_loc_right_y,
                                 cell_loc_left_z,
                                 cell_loc_right_z,
                                 ixyz,
                                 Hydro::IA0);

  uLoc[Hydro::IA1] = ONE_F - uLoc[Hydro::IA0];

  uLoc[Hydro::IE] = extrapol_3d(cell_loc_neigh,
                                cell_loc_left_x,
                                cell_loc_right_x,
                                cell_loc_left_y,
                                cell_loc_right_y,
                                cell_loc_left_z,
                                cell_loc_right_z,
                                ixyz,
                                Hydro::IE);

  uLoc[Hydro::IU] = extrapol_3d(cell_loc_neigh,
                                cell_loc_left_x,
                                cell_loc_right_x,
                                cell_loc_left_y,
                                cell_loc_right_y,
                                cell_loc_left_z,
                                cell_loc_right_z,
                                ixyz,
                                Hydro::IU);

  uLoc[Hydro::IV] = extrapol_3d(cell_loc_neigh,
                                cell_loc_left_x,
                                cell_loc_right_x,
                                cell_loc_left_y,
                                cell_loc_right_y,
                                cell_loc_left_z,
                                cell_loc_right_z,
                                ixyz,
                                Hydro::IV);

  uLoc[Hydro::IW] = extrapol_3d(cell_loc_neigh,
                                cell_loc_left_x,
                                cell_loc_right_x,
                                cell_loc_left_y,
                                cell_loc_right_y,
                                cell_loc_left_z,
                                cell_loc_right_z,
                                ixyz,
                                Hydro::IW);

  // compute primitive variables in current cell
  HydroState<dim>             qLoc; // primitive    variables in current cell
  [[maybe_unused]] const auto c =
    models::computePrimitives<dim>(uLoc, qLoc, m_hydro_settings, m_eos);

  // write primitive variables
  set_primitive_vars(cellindex_out, iOct_out, qLoc);

} // linear_extrapolate_using_limited_slopes - 3d

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::conservative_interpolation_order2(
  CellLocation<dim> const &       cell_loc_neigh,
  coord_t<dim> const &            coord_out,
  index_t const &                 cellindex_out,
  [[maybe_unused]] iOct_t const & iOct_global,
  iOct_t const &                  iOct_out) const
{
  // determine local position of current cell inside virtual parent cell using integer
  //  coordinates in {0, 1}
  const auto ixyz = get_local_position<dim>(coord_out);

  HydroState<dim> uLoc; // conservative variables in current cell

  uLoc[Hydro::IAD0] = interpol_order2(cell_loc_neigh, ixyz, Hydro::IAD0);
  uLoc[Hydro::IAD1] = interpol_order2(cell_loc_neigh, ixyz, Hydro::IAD1);
  uLoc[Hydro::ID] = uLoc[Hydro::IAD0] + uLoc[Hydro::IAD1];
  uLoc[Hydro::IA0] = interpol_order2(cell_loc_neigh, ixyz, Hydro::IA0);
  uLoc[Hydro::IA1] = ONE_F - uLoc[Hydro::IA0];
  uLoc[Hydro::IE] = interpol_order2(cell_loc_neigh, ixyz, Hydro::IE);
  uLoc[Hydro::IU] = interpol_order2(cell_loc_neigh, ixyz, Hydro::IU);
  uLoc[Hydro::IV] = interpol_order2(cell_loc_neigh, ixyz, Hydro::IV);

  if constexpr (dim == 3)
  {
    uLoc[Hydro::IW] = interpol_order2(cell_loc_neigh, ixyz, Hydro::IW);
  }

  // compute primitive variables in current cell
  HydroState<dim>             qLoc; // primitive    variables in current cell
  [[maybe_unused]] const auto c =
    models::computePrimitives<dim>(uLoc, qLoc, m_hydro_settings, m_eos);

  // write primitive variables
  set_primitive_vars(cellindex_out, iOct_out, qLoc);

} // ConvertToPrimitivesVariablesFunctor<dim, device_t>::conservative_interpolation_order2

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::conservative_interpolation_order4(
  CellLocation<dim> const &       cell_loc_neigh,
  coord_t<dim> const &            coord_out,
  index_t const &                 cellindex_out,
  [[maybe_unused]] iOct_t const & iOct_global,
  iOct_t const &                  iOct_out) const
{
  // determine local position of current cell inside virtual parent cell using integer
  //  coordinates in {0, 1}
  const auto ixyz = get_local_position<dim>(coord_out);

  HydroState<dim> uLoc; // conservative variables in current cell

  uLoc[Hydro::IAD0] = interpol_order4(cell_loc_neigh, ixyz, Hydro::IAD0);
  uLoc[Hydro::IAD1] = interpol_order4(cell_loc_neigh, ixyz, Hydro::IAD1);
  uLoc[Hydro::ID] = uLoc[Hydro::IAD0] + uLoc[Hydro::IAD1];
  uLoc[Hydro::IA0] = interpol_order4(cell_loc_neigh, ixyz, Hydro::IA0);
  uLoc[Hydro::IA1] = ONE_F - uLoc[Hydro::IA0];
  uLoc[Hydro::IE] = interpol_order4(cell_loc_neigh, ixyz, Hydro::IE);
  uLoc[Hydro::IU] = interpol_order4(cell_loc_neigh, ixyz, Hydro::IU);
  uLoc[Hydro::IV] = interpol_order4(cell_loc_neigh, ixyz, Hydro::IV);

  if constexpr (dim == 3)
  {
    uLoc[Hydro::IW] = interpol_order4(cell_loc_neigh, ixyz, Hydro::IW);
  }

  // compute primitive variables in current cell
  HydroState<dim>             qLoc; // primitive    variables in current cell
  [[maybe_unused]] const auto c =
    models::computePrimitives<dim>(uLoc, qLoc, m_hydro_settings, m_eos);

  // write primitive variables
  set_primitive_vars(cellindex_out, iOct_out, qLoc);

} // ConvertToPrimitivesVariablesFunctor<dim, device_t>::conservative_interpolation_order4

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::fill_ghosts(index_t const &      cellindex_out,
                                                                coord_t<dim> const & coord_out,
                                                                iOct_t const &       iOct_global,
                                                                iOct_t const &       iOct_out) const
{

  const auto & b = m_userdata_in.block_size();

  // coordinates of source cell (where to read data)
  coord_t<dim> coord_in;
  const auto   dir = ghosted_coords_to_inner_coords(coord_in, coord_out, b);

  int32_t cellindex_in = coord_to_cellindex<dim>(coord_in, m_userdata_in.block_size());

  auto dir_norm = dir[IX] * dir[IX] + dir[IY] * dir[IY];
  if constexpr (dim == 3)
    dir_norm += dir[IZ] * dir[IZ];

  if (dir_norm == 0)
  {
    // current cell is inside current block
    fill_inner(cellindex_in, cellindex_out, iOct_global, iOct_out);
  }
  else
  {
    // current cell is a ghost cell (thus belonging to a neighbor block)

    /*
     * fill actual ghosts with data from a neighbor block.
     */

    // get orchard key of current octant
    auto key_cur = m_stencil_helper.key(iOct_global);

    shift_t<dim> shift;
    shift[IX] = b[IX] * dir[IX];
    shift[IY] = b[IY] * dir[IY];
    if constexpr (dim == 3)
    {
      shift[IZ] = b[IZ] * dir[IZ];
    }

    const CellLocation_t cell_loc_cur{ coord_in, key_cur, iOct_global, false };
    const auto           cell_loc_neigh = m_stencil_helper.getNeighLoc(cell_loc_cur, shift);

    /*
     * Dealing with the 3 possibilities:
     * - neighbor octant is at same    AMR level : doing a simple copy
     * - neighbor octant is at finer   AMR level : doing a restriction (average values)
     * - neighbor octant is at coarser AMR level : doing a prolongation
     */
    if (cell_loc_neigh.level() >= cell_loc_cur.level())
    {
      // doing a simple copy or doing a restriction when neighbor is at higher AMR level
      fill_ghost_copy(cell_loc_cur, cell_loc_neigh, cellindex_out, iOct_out);
    }
    else if (cell_loc_neigh.level() + 1 == cell_loc_cur.level())
    {
      // doing a prolongation because neighbor is coarser

      if (m_prolongation == +CellCenteredProlongationType::SIMPLE_COPY)
      {
        // simple copy of the coarse value
        fill_ghost_copy(cell_loc_cur, cell_loc_neigh, cellindex_out, iOct_out);
      }
      else if (m_prolongation == +CellCenteredProlongationType::EXTRAPOLATE_LINEAR_MINMOD)
      {
        linear_extrapolate_using_limited_slopes(
          cell_loc_neigh, coord_in, iOct_global, cellindex_out, iOct_out);
      }
      else if (m_prolongation == +CellCenteredProlongationType::CONSERVATIVE_INTERPOLATION_ORDER_2)
      {
        conservative_interpolation_order2(
          cell_loc_neigh, coord_out, cellindex_out, iOct_global, iOct_out);
      }
      else if (m_prolongation == +CellCenteredProlongationType::CONSERVATIVE_INTERPOLATION_ORDER_4)
      {
        conservative_interpolation_order4(
          cell_loc_neigh, coord_out, cellindex_out, iOct_global, iOct_out);
      }
      else
      {
        // we shouldn't be here
        KOKKOS_ASSERT(false && "Unsupported prolongation type");
      }
    }
    else
    {
      KOKKOS_ASSERT(false && "Logic error: neighbor octant not found (Kernel Panic !)");
    }

  } // end if (dir_norm ==0)

} // fill_ghosts

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::operator()(TagComputeAllQuad const &,
                                                               const index_t & global_index) const
{

  const auto nbCellsPerGhostedLeaf = m_userdata_out.num_cells();

  // retrieve local octant index (this is where we want to write data)
  const auto iOct_local = global_index / nbCellsPerGhostedLeaf;
  const auto cell_index_out = global_index - iOct_local * nbCellsPerGhostedLeaf;

  // retrieve global octant index
  const auto iOct_global = m_iOct_begin + iOct_local;

  // compute cartesian coordinates inside ghosted block
  const auto coord_out = cellindex_to_coord<dim>(
    cell_index_out, m_userdata_out.ghosted_block_size(), m_userdata_out.shift());

  fill_ghosts(cell_index_out, coord_out, iOct_global, iOct_local);

} // operator() - TagComputeAllQuad

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ConvertToPrimitivesVariablesFunctor<dim, device_t>::operator()(TagComputeMirrorQuad const &,
                                                               const index_t & global_index) const
{

  const auto nbCellsPerGhostedLeaf = m_userdata_out.num_cells();

  // retrieve mirror index
  const auto iMirror = global_index / nbCellsPerGhostedLeaf;
  const auto cell_index_out = global_index - iMirror * nbCellsPerGhostedLeaf;

  // retrieve key associated to that mirror index
  const auto mirror_key = m_mirror_orchard_keys_device(iMirror);

  // make sure the key is in the hashmap and retrieve value
  const auto mirror_hashindex = m_stencil_helper.m_amr_hashmap_device.find(mirror_key);
  [[maybe_unused]] const auto valid =
    m_stencil_helper.m_amr_hashmap_device.valid_at(mirror_hashindex);

  KOKKOS_ASSERT(
    valid && "(mirror quadrant) key doesn't exist in hashmap (this is in principle not possible, "
             "since mirror keys are computed from p4est ghosts.)");

  // retrieve iOct associated to that mirror quadrant
  const auto iOct_global = m_stencil_helper.m_amr_hashmap_device.value_at(mirror_hashindex);

  // compute cartesian coordinates inside ghosted block
  const auto coord_out = cellindex_to_coord<dim>(
    cell_index_out, m_userdata_out.ghosted_block_size(), m_userdata_out.shift());

  fill_ghosts(cell_index_out, coord_out, iOct_global, iMirror);

} // operator() - TagComputeMirrorQuad

// explicit template instantiation
template class ConvertToPrimitivesVariablesFunctor<2, kalypsso::DefaultDevice>;
template class ConvertToPrimitivesVariablesFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso
