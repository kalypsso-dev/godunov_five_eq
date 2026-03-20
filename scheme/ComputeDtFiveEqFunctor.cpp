// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeDtFiveEqFunctor.cpp
 */
#include <godunov_five_eq/scheme/ComputeDtFiveEqFunctor.h>

#include <kalypsso/core/brick_utils.h>
#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{
namespace godunov_five_eq
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
ComputeDtFiveEqFunctor<dim, device_t>::ComputeDtFiveEqFunctor(
  ConfigMap const &         config_map,
  orchard_key_view_t        orchard_keys,
  int32_t                   local_num_octants,
  HydroSettings             hydro_settings,
  FieldMap<models::FiveEq>  fm,
  block_size_t<dim>         block_sizes,
  DataArrayBlock_t          Udata,
  eos::EosWrapper<device_t> eos,
  bool                      gravity_enabled,
  UniformGravityField<dim>  gravity_field)
  : m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_hydro_settings(hydro_settings)
  , m_fm(fm)
  , m_block_sizes(block_sizes)
  , m_nbCellsPerLeaf(Udata.num_cells())
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_Udata(Udata)
  , m_eos(eos)
  , m_gravity_enabled(gravity_enabled)
  , m_gravity_field(gravity_field)
{}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
ComputeDtFiveEqFunctor<dim, device_t>::apply(ConfigMap const &         config_map,
                                             orchard_key_view_t        orchard_keys,
                                             int32_t                   local_num_octants,
                                             HydroSettings             hydro_settings,
                                             FieldMap<models::FiveEq>  fm,
                                             block_size_t<dim>         block_sizes,
                                             DataArrayBlock_t          Udata,
                                             eos::EosWrapper<device_t> eos,
                                             real_t &                  invDt)
{
  const auto gravity_enabled = config_map.getBool("gravity", "enabled", false);
  const auto gravity_field = get_uniform_gravity_vector<dim>(config_map);

  ComputeDtFiveEqFunctor functor(config_map,
                                 orchard_keys,
                                 local_num_octants,
                                 hydro_settings,
                                 fm,
                                 block_sizes,
                                 Udata,
                                 eos,
                                 gravity_enabled,
                                 gravity_field);

  Kokkos::Max<real_t> reducer(invDt);

  const auto nbCellsPerLeaf = Udata.num_cells();

  // compute total number of cells
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_reduce("kalypsso::godunov_five_eq::ComputeDtFiveEqFunctor",
                          Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                          functor,
                          reducer);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeDtFiveEqFunctor<dim, device_t>::compute_cfl(int32_t const & iOct,
                                                   int32_t const & cell_index,
                                                   real_t &        invDt) const
{
  HydroState<dim>            uLoc; // conservative variables in current cell
  HydroState<dim>            qLoc; // primitive    variables in current cell
  Kokkos::Array<real_t, dim> v;    // velocity

  // get block level
  const auto level = orchard_key_t<dim>::level(m_orchard_keys(iOct));

  // compute cell size (assume dx=dy=dz, i.e. block_sizes are the same along all directions)
  const auto dx = compute_cell_length<dim>(level, m_block_sizes[IX]) * m_scaling_factor;

  // get conservative variable in current cell
  uLoc[FiveEq::ID0] = m_Udata(cell_index, m_fm[FiveEq::ID0], iOct);
  uLoc[FiveEq::ID1] = m_Udata(cell_index, m_fm[FiveEq::ID1], iOct);
  uLoc[FiveEq::ID] = uLoc[FiveEq::ID0] + uLoc[FiveEq::ID1];
  uLoc[FiveEq::IPHI] = m_Udata(cell_index, m_fm[FiveEq::IPHI], iOct);
  uLoc[FiveEq::IE] = m_Udata(cell_index, m_fm[FiveEq::IE], iOct);
  uLoc[FiveEq::IU] = m_Udata(cell_index, m_fm[FiveEq::IU], iOct);
  uLoc[FiveEq::IV] = m_Udata(cell_index, m_fm[FiveEq::IV], iOct);
  if constexpr (dim == 3)
    uLoc[FiveEq::IW] = m_Udata(cell_index, m_fm[FiveEq::IW], iOct);

  // get primitive variables and mixture speed of sound in current cell
  const auto c = models::computePrimitives(uLoc, qLoc, m_hydro_settings, m_eos);

  // compute velocity
  v[IX] = c + fabs(qLoc[FiveEq::IU]);
  v[IY] = c + fabs(qLoc[FiveEq::IV]);
  if constexpr (dim == 3)
    v[IZ] = c + fabs(qLoc[FiveEq::IW]);

  // update cfl
  if constexpr (dim == 2)
    invDt = fmax(invDt, v[IX] / dx + v[IY] / dx);

  else if constexpr (dim == 3)
    invDt = fmax(invDt, v[IX] / dx + v[IY] / dx + v[IZ] / dx);

} // compute_cfl

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeDtFiveEqFunctor<dim, device_t>::compute_cfl_with_gravity(int32_t const & iOct,
                                                                int32_t const & cell_index,
                                                                real_t &        invDt) const
{
  HydroState<dim> uLoc; // conservative variables in current cell
  HydroState<dim> qLoc; // primitive    variables in current cell

  // get block level
  const auto level = orchard_key_t<dim>::level(m_orchard_keys(iOct));

  // compute cell size (assume dx=dy=dz, i.e. block_sizes are the same along all directions)
  const auto dx = compute_cell_length<dim>(level, m_block_sizes[IX]) * m_scaling_factor;

  // get conservative variable in current cell
  uLoc[FiveEq::ID0] = m_Udata(cell_index, m_fm[FiveEq::ID0], iOct);
  uLoc[FiveEq::ID1] = m_Udata(cell_index, m_fm[FiveEq::ID1], iOct);
  uLoc[FiveEq::ID] = uLoc[FiveEq::ID0] + uLoc[FiveEq::ID1];
  uLoc[FiveEq::IPHI] = m_Udata(cell_index, m_fm[FiveEq::IPHI], iOct);
  uLoc[FiveEq::IE] = m_Udata(cell_index, m_fm[FiveEq::IE], iOct);
  uLoc[FiveEq::IU] = m_Udata(cell_index, m_fm[FiveEq::IU], iOct);
  uLoc[FiveEq::IV] = m_Udata(cell_index, m_fm[FiveEq::IV], iOct);
  if constexpr (dim == 3)
    uLoc[FiveEq::IW] = m_Udata(cell_index, m_fm[FiveEq::IW], iOct);

  // get primitive variables and mixture speed of sound in current cell
  const auto c = models::computePrimitives(uLoc, qLoc, m_hydro_settings, m_eos);

  real_t velocity = ZERO_F;
  velocity += c + fabs(qLoc[FiveEq::IU]);
  velocity += c + fabs(qLoc[FiveEq::IV]);
  if constexpr (dim == 3)
  {
    velocity += c + fabs(qLoc[FiveEq::IW]);
  }

  /* Due to the gravitational acceleration, the CFL condition
   * can be written as
   * g dt^2 / (2 dx) + u dt / dx <= cfl
   * where u = sum(|v_i| + c_s) and g = sum(|g_i|)
   *
   * u / dx has to be corrected by a factor k / (sqrt(1 + 2k) - 1)
   * in order to satisfy the new CFL, where k = g dx cfl / u^2
   */
  real_t k = fabs(m_gravity_field[IX]) + fabs(m_gravity_field[IY]);
  if constexpr (dim == 3)
  {
    k += fabs(m_gravity_field[IZ]);
  }

  k *= dx / (velocity * velocity);

  // prevent numerical errors due to very low gravity
  k = fmax(k, KALYPSSO_NUM(1e-4));

  velocity *= k / (sqrt(ONE_F + TWO_F * k) - ONE_F);

  // update cfl
  invDt = fmax(invDt, velocity / dx);

} // compute_cfl_with_gravity

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeDtFiveEqFunctor<dim, device_t>::operator()(const index_t & global_index,
                                                  real_t &        invDt) const
{
  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_nbCellsPerLeaf;
  const auto cell_index = global_index - iOct * m_nbCellsPerLeaf;

  // compute cfl in current cell and update invDt
  if (m_gravity_enabled)
  {
    compute_cfl_with_gravity(iOct, cell_index, invDt);
  }
  else
  {
    compute_cfl(iOct, cell_index, invDt);
  }
} // operator()

// explicit template instantiation
template class ComputeDtFiveEqFunctor<2, kalypsso::DefaultDevice>;
template class ComputeDtFiveEqFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso
