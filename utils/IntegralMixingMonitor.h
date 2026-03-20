// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file IntegralMixingMonitor.h
 *
 * \brief Declare class used to compute volume integral of mixing and record value for time history.
 */

#ifndef KALYPSSO_GODUNOV_FIVE_EQ_INTEGRAL_MIXING_MONITOR_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_INTEGRAL_MIXING_MONITOR_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h> // for math functions (max, min, ...)
#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

namespace kalypsso
{

namespace godunov_five_eq
{

class MixingParams
{
private:
  //! Physical time when run starts
  real_t m_t_start;

  //! Physical time when run stops
  real_t m_t_end;

  //! Total number of records requested
  int32_t m_num_records;

  //! Time interval between two consecutive records
  real_t m_interval;

  //! Current number of recorded values
  int32_t m_recorded;

  //! file prefix
  std::string m_file_prefix;

public:
  MixingParams(ConfigMap const & config_map)
    : m_t_start(config_map.getReal("run", "tStart", KALYPSSO_NUM(0.0)))
    , m_t_end(config_map.getReal("run", "tEnd", KALYPSSO_NUM(0.0)))
    , m_num_records(config_map.getInteger("mixing", "num_records", 10))
    , m_interval((m_t_end - m_t_start) / static_cast<real_t>(m_num_records))
    , m_recorded(0)
    , m_file_prefix(config_map.getString("output", "outputPrefix", "output"))
  {
    // TODO: we will need to adjust tStart when we will restart a restart run
  }

  const auto
  num_records() const
  {
    return m_num_records;
  }

  const int32_t &
  recorded() const
  {
    return m_recorded;
  }

  int32_t &
  recorded()
  {
    return m_recorded;
  }

  auto
  file_prefix() const
  {
    return m_file_prefix;
  }

  /**
   * Run true when current time is close enough to a value that need to be recorded.
   *
   * \param[in] iter Time iterations.
   * \param[in] time Physical time.
   */
  bool
  should_record(int32_t iter, real_t tCurrent) const
  {
    // num_records == 0 means no output at all
    if (m_num_records == 0)
    {
      return false;
    }

    // num_records < 0  means always output
    if (m_num_records < 0)
    {
      return true;
    }

    if (iter == 0)
    {
      return true;
    }
    else if (tCurrent > static_cast<real_t>(m_recorded + 1) * m_interval)
    {
      return true;
    }

    // always write the last time step
    if (ISFUZZYNULL(tCurrent - m_t_end))
    {
      return true;
    }

    return false;
  } // should_record

}; // struct MixingParams

/**
 * \class IntegralMixingMonitor
 * \brief Compute volume integral of mixing and record value for time history.
 *
 *
 * \tparam dim The dimension of the problem (must be 2 or 3).
 * \tparam device_t On which Kokkos device to run the underlying functor.
 *
 */
template <size_t dim, typename device_t>
class IntegralMixingMonitor
{
public:
  using OrchardKeys = typename orchard_key_base_t<device_t>::view_t;

  /**
   * Constructor.
   *
   * \param[in] config_map Inputted config map.
   */
  IntegralMixingMonitor(const ConfigMap & config_map)
    : m_mixing_params(config_map)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_mixings("mixing", static_cast<size_t>(m_mixing_params.num_records()))
    , m_times("times", static_cast<size_t>(m_mixing_params.num_records()))
  {}

  /**
   * \brief Computes volume integral of mixing and store the result.
   *
   * Mixing is defined as phi*(1-phi) where phi is a volume fraction; so it should be non zero only
   * near material interface.
   *
   * \param[in] data Values array of the simulation.
   * \param[in] keys Orchard keys.
   * \param[in] start_octant The first octant to compute.
   * \param[in] end_octant The last octant to compute, excluded.
   * \param[in] i_phi Index to access volume fraction.
   * \param[in] current_time Current time in physical units.
   * \param[in] par_env Parallel environment.
   *
   * \note par_env is only used when the number of MPI processes is strictly larger than 1.
   *
   * \returns The volume integral value.
   */
  void
  record(const DataArrayBlock<dim, real_t, device_t> & data,
         const OrchardKeys &                           keys,
         const int32_t                                 start_octant,
         const int32_t                                 end_octant,
         int32_t                                       i_phi,
         real_t                                        current_time,
         [[maybe_unused]] const ParallelEnv &          par_env);

  /**
   * Run true when current time is close enough to a value that need to be recorded.
   *
   * \param[in] iter Time iterations.
   * \param[in] time Physical time.
   */
  bool
  should_record(int32_t iter, real_t tCurrent) const
  {
    return m_mixing_params.should_record(iter, tCurrent);
  }

  //! save recorded values to file using numpy format
  void
  save();

  //! is this monitoring enabled ?
  bool
  enabled() const
  {
    return m_mixing_params.num_records() != 0;
  }

private:
  //! Kokkos execution space
  using ExecutionSpace = typename device_t::execution_space;

  //! Mixing parameters
  MixingParams m_mixing_params;

  //! The variable index to use to access volume fraction

  //! Tree scaling factor (used for computing local metric)
  real_t m_scaling_factor;

  //! Array of mixing values
  Kokkos::View<real_t *, Kokkos::HostSpace> m_mixings;

  //! Array of times values
  Kokkos::View<real_t *, Kokkos::HostSpace> m_times;

}; // class IntegralMixingMonitor

extern template class IntegralMixingMonitor<2, kalypsso::DefaultDevice>;
extern template class IntegralMixingMonitor<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_INTEGRAL_MIXING_MONITOR_H_
