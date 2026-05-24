// SPDX-FileCopyrightText: 2026 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FixVolumeFractionsNormalization.cpp
 */
#include <godunov_five_eq/scheme/FixVolumeFractionsNormalization.h>

namespace kalypsso
{

namespace godunov_five_eq
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim, typename device_t>
FixVolumeFractionsNormalization<dim, device_t>::FixVolumeFractionsNormalization(
  DataArrayBlock_t const & U)
  : m_U(U)
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
FixVolumeFractionsNormalization<dim, device_t>::apply(AMRMeshInfo const &      amr_mesh_info,
                                                      DataArrayBlock_t const & U)
{

  FixVolumeFractionsNormalization<dim, device_t> functor(U);

  // number of owned quadrant x number of block cells
  const auto nbIterations = amr_mesh_info.local_num_quadrants() * U.num_cells();

  // Update all variables
  Kokkos::parallel_for("kalypsso::godunov_five_eq::FixVolumeFractionsNormalization",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FixVolumeFractionsNormalization<dim, device_t>::operator()(const index_t & global_index) const
{

  const auto num_cells = m_U.num_cells();

  // retrieve local octant index
  auto const iOct_local = global_index / num_cells;
  auto const cell_index = global_index - iOct_local * num_cells;

  m_U(cell_index, Hydro::IA1, iOct_local) = ONE_F - m_U(cell_index, Hydro::IA0, iOct_local);
  m_U(cell_index, Hydro::ID, iOct_local) =
    m_U(cell_index, Hydro::IAD0, iOct_local) + m_U(cell_index, Hydro::IAD1, iOct_local);

} // operator ()

// explicit template instantiation
template class FixVolumeFractionsNormalization<2, kalypsso::DefaultDevice>;
template class FixVolumeFractionsNormalization<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso
