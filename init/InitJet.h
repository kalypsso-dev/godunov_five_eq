// SPDX-FileCopyrightText: 2026 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitJet.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_INIT_JET_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_INIT_JET_H_

#include <godunov_five_eq/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve jet testcase.
 *
 * Init: fluid at rest in a mesh uniformly refined at level min.
 *
 * - material 0 is the material at rest filling the domain (region 0)
 * - material 1 is the material injected into the domain (region 1)
 */
template <size_t dim, typename device_t>
class InitJetDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! this test case has 3 regions
  static constexpr int NB_REGIONS = 2;

private:
  //! heavy data
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! Initial states (one per region, conservative variables)
  InitialStates<dim, device_t> m_initial_states;

  //! Number of materials
  int m_num_materials;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitJetDataFunctor(DataArrayBlock_t const &             Udata,
                     orchard_key_view_t<device_t> const & orchard_keys,
                     int32_t                              local_num_octants,
                     InitialStates<dim, device_t> const & initial_states,
                     ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_initial_states(initial_states)
    , m_num_materials(config_map.getInteger("run", "nmat", 0))
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

public:
  //! static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        orchard_key_view_t<device_t> const & orchard_keys,
        int32_t                              local_num_octants,
        InitialStates<dim, device_t> const & initial_states,
        ConfigMap const &                    config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int32_t & global_index) const;

}; // InitJetDataFunctor

extern template class InitJetDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitJetDataFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
// =======================================================
/**
 * Hydrodynamical jet shock tube init.
 */
template <size_t dim, typename device_t>
class InitJet
{
public:
  static void
  apply(SolverGodunovFiveEq<dim, device_t> & solver);
}; // class InitJet

extern template class InitJet<2, kalypsso::DefaultDevice>;
extern template class InitJet<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_INIT_JET_H_
