// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitRichtmyerMeshkov.h
 *
 * References:
 * - https://doi.org/10.1016/j.compfluid.2021.105158 : A volume-of-fluid reconstruction based
 * interface sharpening algorithm for a reduced equation model of two-material compressible
 * flow, M. Cutforth et al., Computers & Fluids Volume 231, 15 December 2021, 105158
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_INIT_RICHTMYER_MESHKOV_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_INIT_RICHTMYER_MESHKOV_H_

#include <godunov_five_eq/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/RichtmyerMeshkovParams.h>
#include <kalypsso/core/mesh_utils.h>
#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve Richtmyer-Meshkov instability test case.
 *
 * \sa references
 * - https://doi.org/10.1016/j.compfluid.2021.105158
 *
 * Initial conditions is refined near strong density gradients.
 */
template <size_t dim, typename device_t>
class InitRichtmyerMeshkovDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! a Richtmyer-Meshkov problem has 3 regions (1 post-shock + 2 pre-shock)
  static constexpr int NB_REGIONS = 3;

private:
  //! heavy data
  DataArrayBlock_t m_Udata;

  //! field manager
  FieldMap<models::FiveEq> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! RichtmyerMeshkov problem specific parameters (used on device)
  RichtmyerMeshkovParams m_rm_params;

  //! Initial states (one per region, conservative variables)
  InitialStates<dim, device_t> m_initial_states;

  //! Number of materials
  int m_num_materials;

  //! Equation of state wrapper
  EosWrapper_t<device_t> m_eos_wrapper;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitRichtmyerMeshkovDataFunctor(DataArrayBlock_t const &             Udata,
                                  FieldMap<models::FiveEq>             fm,
                                  orchard_key_view_t<device_t> const & orchard_keys,
                                  int32_t                              local_num_octants,
                                  InitialStates<dim, device_t> const & initial_states,
                                  ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_fm(fm)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_rm_params(config_map)
    , m_initial_states(initial_states)
    , m_num_materials(config_map.getInteger("run", "nmat", 0))
    , m_eos_wrapper(config_map)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

public:
  //! static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        FieldMap<models::FiveEq>             fm,
        orchard_key_view_t<device_t> const & orchard_keys,
        int32_t                              local_num_octants,
        InitialStates<dim, device_t> const & initial_states,
        ConfigMap const &                    config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int32_t & global_index) const;

private:
  // ==========================================================================
  // ==========================================================================
  /**
   * Utility to determine in which region a given point is.
   */
  KOKKOS_INLINE_FUNCTION int
  point_to_region(Kokkos::Array<real_t, dim> const & xyz) const
  {
    const int region_id = xyz[IX] >= m_rm_params.x_shock               ? 0  // material 0
                          : xyz[IX] >= m_rm_params.x_material(xyz[IY]) ? 1  // material 0
                                                                       : 2; // material 1
    return region_id;
  }

  /**
   * For each corner of a given cell, determine in which region this corner is.
   *
   * \param[in] ijk i-j-k indexes of a cell inside a block of cells
   * \param[in] key orchard key of current octant
   * \param[out] regions array of regions (one per corner)
   */
  KOKKOS_INLINE_FUNCTION void
  compute_corner_to_region(coord_t<dim> const &                             ijk,
                           key_t const &                                    key,
                           Kokkos::Array<int, Corner::num_corners<dim>()> & regions) const
  {
    const auto & block_sizes = m_Udata.block_size();

    for (uint8_t i_corner = 0; i_corner < Corner::num_corners<dim>(); i_corner++)
    {
      const auto xyz_vertex_corner =
        orchard_key_to_corner_coord<dim>(key, ijk, block_sizes[IX], i_corner);
      auto xyz_corner =
        vertex_coord_to_real_space<dim>(xyz_vertex_corner, m_scaling_factor, m_xyz_min);

      regions[i_corner] = point_to_region(xyz_corner);
    }
  }

}; // InitRichtmyerMeshkovDataFunctor

extern template class InitRichtmyerMeshkovDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitRichtmyerMeshkovDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve RichtmyerMeshkov problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitRichtmyerMeshkovDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitRichtmyerMeshkovRefineFunctor
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! type alias for a (device) Kokkos view of refinement flags
  using amrflags_view_t = typename AMRContext<dim, device_t>::amrflags_view_t;

  struct TagRefineAlways
  {};
  struct TagRefineGeometric
  {};

private:
  //! heavy hydrodynamics data
  DataArrayBlock_t m_Udata;

  //! field manager
  FieldMap<models::FiveEq> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! RichtmyerMeshkov problem specific parameters (used on device)
  RichtmyerMeshkovParams m_rm_params;

  //! which level should we look at
  int m_level_refine;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitRichtmyerMeshkovRefineFunctor(DataArrayBlock_t const &             Udata,
                                    FieldMap<models::FiveEq>             fm,
                                    orchard_key_view_t<device_t> const & orchard_keys,
                                    amrflags_view_t const &              amrflags,
                                    int32_t                              local_num_octants,
                                    int                                  level_refine,
                                    ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_fm(fm)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_rm_params(config_map)
    , m_level_refine(level_refine)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

public:
  //! static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        FieldMap<models::FiveEq>             fm,
        orchard_key_view_t<device_t> const & orchard_keys,
        amrflags_view_t const &              amrflags,
        int32_t                              local_num_octants,
        int                                  level_refine,
        ConfigMap const &                    config_map);

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineAlways const &, const size_t & iOct) const;

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineGeometric const &, const size_t & iOct) const;

}; // class InitRichtmyerMeshkovRefineFunctor

extern template class InitRichtmyerMeshkovRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitRichtmyerMeshkovRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * \class InitRichtmyerMeshkov
 *
 * Richtmyer-Meshkov instability  init.
 */
template <size_t dim, typename device_t>
class InitRichtmyerMeshkov
{
public:
  static void
  apply(SolverGodunovFiveEq<dim, device_t> & solver);

}; // class InitRichtmyerMeshkov

extern template class InitRichtmyerMeshkov<2, kalypsso::DefaultDevice>;
extern template class InitRichtmyerMeshkov<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_INIT_RICHTMYER_MESHKOV_H_
