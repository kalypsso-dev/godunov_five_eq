// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitJet.cpp
 */

#include <godunov_five_eq/init/InitJet.h>
#include <godunov_five_eq/SolverGodunovFiveEq.h>
#include <godunov_five_eq/models/utils.h>

#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{
namespace godunov_five_eq
{
// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitJetDataFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                         orchard_key_view_t<device_t> const & orchard_keys,
                                         int32_t                              local_num_octants,
                                         InitialStates<dim, device_t> const & initial_states,
                                         ConfigMap const &                    config_map)
{
  // data init functor
  InitJetDataFunctor functor(Udata, orchard_keys, local_num_octants, initial_states, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_five_eq::InitJetDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitJetDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitJetDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  using Hydro = models::FiveEq<dim>;

  // initialize
  // region 0: material 0 only
  m_Udata(cell_index, Hydro::IAD0, iOct) = m_initial_states(0)[Hydro::IAD0];
  m_Udata(cell_index, Hydro::IAD1, iOct) = m_initial_states(0)[Hydro::IAD1];
  m_Udata(cell_index, Hydro::ID, iOct) =
    m_Udata(cell_index, Hydro::IAD0, iOct) + m_Udata(cell_index, Hydro::IAD1, iOct);

  m_Udata(cell_index, Hydro::IA0, iOct) = m_initial_states(0)[Hydro::IA0];
  m_Udata(cell_index, Hydro::IA1, iOct) = m_initial_states(0)[Hydro::IA1];
  m_Udata(cell_index, Hydro::IU, iOct) = m_initial_states(0)[Hydro::IU];
  m_Udata(cell_index, Hydro::IV, iOct) = m_initial_states(0)[Hydro::IV];
  if constexpr (dim == 3)
    m_Udata(cell_index, Hydro::IW, iOct) = m_initial_states(0)[Hydro::IW];
  m_Udata(cell_index, Hydro::IE, iOct) = m_initial_states(0)[Hydro::IE];

} // end InitJetDataFunctor::operator ()

template class InitJetDataFunctor<2, kalypsso::DefaultDevice>;
template class InitJetDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitJet<dim, device_t>::apply(SolverGodunovFiveEq<dim, device_t> & solver)
{

  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();

  const auto nb_regions = InitJetDataFunctor<dim, device_t>::NB_REGIONS;
  auto       initial_states = get_initial_states<dim, device_t>(config_map, nb_regions);

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitJetDataFunctor<dim, device_t>::apply(solver.U(),
                                           solver.mesh_map()->orchard_keys(),
                                           solver.amr_mesh()->local_num_quadrants(),
                                           initial_states,
                                           config_map);

#ifdef KALYPSSO_CORE_USE_MPI
  // load balancing (= repartitioning) the octree mesh + userdata over the MPI processes.
  // U and U2 will be resized
  solver.do_load_balancing();
#endif

} // InitJet::apply

template class InitJet<2, kalypsso::DefaultDevice>;
template class InitJet<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso
