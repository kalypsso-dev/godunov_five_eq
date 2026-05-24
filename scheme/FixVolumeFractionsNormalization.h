// SPDX-FileCopyrightText: 2026 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FixVolumeFractionsNormalization.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_FIX_VOLUME_FRACTIONS_NORMALIZATION_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_FIX_VOLUME_FRACTIONS_NORMALIZATION_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/AMRMeshInfo.h>

// model
#include <godunov_five_eq/models/FiveEq.h>

namespace kalypsso
{

namespace godunov_five_eq
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 *  Fix volume fractions to ensure the normalization property: \f$ sum_i \alpha_i = 1 \f$
 */
template <size_t dim, typename device_t>
class FixVolumeFractionsNormalization
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using Hydro = models::FiveEq<dim>;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

private:
  //! conservative variables - entire mesh - input and output
  DataArrayBlock_t m_U;

public:
  /**
   * Perform time integration (MUSCL Godunov).
   *
   * \param[in]  time step (as computed by CFL condition)
   *
   */
  FixVolumeFractionsNormalization(DataArrayBlock_t const & U);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in a group of octant
  static void
  apply(AMRMeshInfo const & amr_mesh_info, DataArrayBlock_t const & U);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // FixVolumeFractionsNormalization

// explicit template instantiation
extern template class FixVolumeFractionsNormalization<2, kalypsso::DefaultDevice>;
extern template class FixVolumeFractionsNormalization<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_FIX_VOLUME_FRACTIONS_NORMALIZATION_H_
