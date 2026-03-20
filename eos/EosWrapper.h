// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file EosWrapper.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_MODELS_EOS_WRAPPER_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_MODELS_EOS_WRAPPER_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/eos/IdealGasEos.h>
#include <kalypsso/core/eos/IdealGasMixture.h>
#include <kalypsso/core/eos/StiffenedGasEos.h>
#include <kalypsso/core/eos/StiffenedGasMixture.h>
#include <kalypsso/core/eos/eos_utils.h>

namespace kalypsso
{

namespace godunov_five_eq
{

namespace eos
{

// ==========================================================================
// ==========================================================================
/**
 * This a front-end helper to access equation of states data.
 *
 * \todo See if we could refactor using static polymorphism.
 */
template <typename device_t>
class EosWrapper
{
public:
  EosWrapper() = default;

  //! Initialized Eos for a given material by reading property from ini file.
  //!
  //! by default use 2 materials
  EosWrapper(ConfigMap const & config_map)
    : m_eos_type(core::eos::get_eos_type(config_map))
    , m_ig_mixture(config_map)
    , m_sg_mixture(config_map)
  {}

  /**
   * Compute mixture pressure.
   *
   * To be used only when there are two materials
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  mixture_pressure(real_t rho, real_t eint, real_t phi0) const
  {
    if (m_eos_type == +core::eos::EOS_TYPE::IDEAL_GAS)
    {
      return m_ig_mixture.mixture_pressure(rho, eint, phi0);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::STIFFENED_GAS)
    {
      return m_sg_mixture.mixture_pressure(rho, eint, phi0);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::MIE_GRUNEISEN)
    {
      // TODO
    }
    return ZERO_F;
  }

  /**
   * Compute mixture specific internal energy.
   *
   * To be used only when there are two materials
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  mixture_specific_eint(real_t pressure, real_t rho, real_t phi0) const
  {
    if (m_eos_type == +core::eos::EOS_TYPE::IDEAL_GAS)
    {
      return m_ig_mixture.mixture_specific_eint(pressure, rho, phi0);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::STIFFENED_GAS)
    {
      return m_sg_mixture.mixture_specific_eint(pressure, rho, phi0);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::MIE_GRUNEISEN)
    {
      // TODO
    }
    return ZERO_F;
  }

  /**
   * Compute mixture speed of sound.
   *
   * To be used only when there are two materials
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  mixture_sound_speed(real_t pressure, real_t rho, real_t phi0) const
  {
    if (m_eos_type == +core::eos::EOS_TYPE::IDEAL_GAS)
    {
      return m_ig_mixture.mixture_sound_speed(pressure, rho, phi0);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::STIFFENED_GAS)
    {
      return m_sg_mixture.mixture_sound_speed(pressure, rho, phi0);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::MIE_GRUNEISEN)
    {
      // TODO
    }
    return ZERO_F;
  }

private:
  //! type of equation of state
  core::eos::EOS_TYPE m_eos_type;

  //! Ideal gas mixture
  core::eos::IdealGasMixture<device_t> m_ig_mixture;

  //! Stiffened gas mixture
  core::eos::StiffenedGasMixture<device_t> m_sg_mixture;

}; // class EosWrapper

} // namespace eos

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_MODELS_EOS_WRAPPER_H_
