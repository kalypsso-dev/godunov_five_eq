// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeCurvatureAverage.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_COMPUTE_CURVATURE_AVERAGE_FUNCTOR_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_COMPUTE_CURVATURE_AVERAGE_FUNCTOR_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

// hydro utils (conservative versus primitive variable, equation of state, ...)
#include <kalypsso/core/models/HydroState.h>
#include <kalypsso/core/models/utils_hydro.h>
#include <kalypsso/core/utils_block.h>

// other utilities
#ifdef KALYPSSO_CORE_USE_CNPY
#  include <kalypsso/core/cnpy_io.h>
#endif

namespace kalypsso
{
namespace godunov_five_eq
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute curvature average near mixed region.
 *
 * This functor is mainly aimed at checking numerical result of the static droplet test case.
 *
 * \tparam dim is space dimension (2 or 3)
 * \tparam device_t is the Kokkos device use for computation (CPU, GPU, ...)
 */
template <size_t dim, typename device_t>
class ComputeCurvatureAverage
{

public:
  //! type alias for a data array at block level (see kalypsso_data_container.h)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! type alias for a (device) Kokkos view of orchard keys
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  // makes enum Hydro::VarId available
  using FiveEq = models::FiveEq;

  //! global cell index
  using index_t = int32_t;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! field manager
  const FieldMap<models::FiveEq> m_fm;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  //! heavy data - conservative variables
  DataArrayBlock_t m_Udata;

public:
  ComputeCurvatureAverage(ConfigMap const &              config_map,
                          orchard_key_view_t             orchard_keys,
                          int32_t                        local_num_octants,
                          FieldMap<core::models::FiveEq> fm,
                          block_size_t<dim>              block_sizes,
                          DataArrayBlock_t               Udata)
    : m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_fm(fm)
    , m_xyz_min(get_xyz_min<dim>(config_map))
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_Udata(Udata){};

  // ====================================================================
  // ====================================================================
  //! static method which does it all: create and execute functor using range policy
  //!
  //! \param[in] orchard_keys is a vector of all local (owned+ghost) octant orchard/morton keys
  //! \param[in] local_num_octants is the number of octants owned by current MPI process (ghost
  //!            excluded)
  //! \param[in] fm is the field map (TODO refactor this)
  //! \param[in] block_sizes is an array the cartesian block sizes
  //! \param[in,out] invDt is the inverse of time step, the output of this functor
  //!
  static void
  apply([[maybe_unused]] ParallelEnv const & par_env,
        ConfigMap const &                    config_map,
        orchard_key_view_t                   orchard_keys,
        int32_t                              local_num_octants,
        FieldMap<models::FiveEq>             fm,
        DataArrayBlock_t                     Udata)
  {
    ComputeCurvatureAverage functor(config_map, orchard_keys, local_num_octants, fm, Udata);

    real_t              sum_curvature, total_sum_curvature = ZERO_F;
    Kokkos::Sum<real_t> reducer(sum_curvature);

    Kokkos::parallel_for("kalypsso::godunov_five_eq::ComputeCurvatureAverage",
                         Kokkos::RangePolicy<exec_space>(0, Udata.num_quadrants()),
                         functor);

#ifdef KALYPSSO_CORE_USE_MPI
    par_env.comm().MPI_Allreduce<MpiComm::SUM>(&sum_curvature, &total_sum_curvature, 1);
#else
    total_sum_curvature = sum_curvature;
#endif // KALYPSSO_CORE_USE_MPI

  } // apply

  // ====================================================================
  // ====================================================================
  /**
   * range policy functor for computing CFL condition.
   *
   * \param[in] global_index spans range from 0 to nbCellsPerLeaf * local_num_octants-1
   *            (i.e. total number of cells in current MPI process)
   * \param[in,out] invDt is the reduced variable to update
   */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index, real_t & sum_curvature) const
  {
    // convert global index into
    // - octant id
    // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
    const auto iOct = global_index / m_Udata.num_cells();
    const auto cell_index = global_index - iOct * m_Udata.num_cells();

    // compute ix,iy,iz of local cell inside
    // block from index
    const auto iCoord = cellindex_to_coord<dim>(cell_index, m_Udata.block_size());

    // get block orchard key
    const auto key = m_orchard_keys(iOct);

    // compute physical x,y,z for that cell (cell center)
    const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, m_block_sizes[IX]);
    const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  } // operator()

}; // ComputeCurvatureAverage

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_COMPUTE_CURVATURE_AVERAGE_FUNCTOR_H_
