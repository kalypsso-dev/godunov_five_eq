// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file utils.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_MODELS_UTILS_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_MODELS_UTILS_H_

#include <type_traits>

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/HydroParams.h>
#include <godunov_five_eq/models/FiveEq.h>
#include <godunov_five_eq/models/HydroState.h>
#include <kalypsso/core/models/HydroSettings.h>
#include <godunov_five_eq/eos/eos_utils.h>

namespace kalypsso
{

namespace godunov_five_eq
{

namespace models
{

// ====================================================================
// ====================================================================
/**
 * Sanitizing hydro state from small volume fraction.
 *
 * \todo evaluate if 1e-12 is a good value
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION void
hydro_sanitizer(HydroState<dim> & q)
{
  constexpr real_t eps = KALYPSSO_NUM(1e-12);

  // sanitizing value
  if (q[FiveEq::ID0] < eps)
  {
    q[FiveEq::ID0] = ZERO_F;
  }

  if (q[FiveEq::ID1] < eps)
  {
    q[FiveEq::ID1] = ZERO_F;
  }

  if (q[FiveEq::IPHI] < eps)
  {
    q[FiveEq::IPHI] = ZERO_F;
  }

  if (q[FiveEq::IPHI] > ONE_F - eps)
  {
    q[FiveEq::IPHI] = ONE_F;
  }

} // hydro_sanitizer

// ====================================================================
// ====================================================================
/**
 * Ensure volume fraction is in range [0, 1]
 *
 * \param[in] phi is volumic fraction
 */
KOKKOS_INLINE_FUNCTION
auto
sanitize_phi(real_t phi)
{
  return fmax(ZERO_F, fmin(ONE_F, phi));
}

// ====================================================================
// ====================================================================
/**
 *
 */
KOKKOS_INLINE_FUNCTION
real_t
sanitize_dphi(real_t phi, real_t dphi)
{
  const auto phi_new = fmax(KALYPSSO_NUM(0.0), fmin(KALYPSSO_NUM(1.0), phi + dphi));
  return phi_new - phi;
}

// ====================================================================
// ====================================================================
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_valid_primitive_state(HydroState<dim> const & q)
{
  const auto   rho = q[FiveEq::ID0] + q[FiveEq::ID1];
  auto const & P = q[FiveEq::IP];
  auto const & phi = q[FiveEq::IPHI];

  return (q[FiveEq::ID0] >= 0 and q[FiveEq::ID1] >= 0 and P >= 0 and phi >= ZERO_F and
          phi <= ONE_F);

} // is_valid_primitive_state

// ====================================================================
// ====================================================================
/**
 * Convert conservative variables (phi0rho0, phi1rho1, phi , rho*u, rho*v, rho*w, e) to
 * primitive variables (rho,u,v,w,p)
 * \param[in]  u  conservative variables array (bifluid state)
 * \param[out] q  primitive    variables array (single fluid equivalent)
 *
 * \note for simplicity we use the same class HydroState for both conservative and primitive
 * variables even though primitive variable state is smaller.
 *
 * \return c local mixture speed of sound
 */
template <typename device_t>
KOKKOS_INLINE_FUNCTION real_t
computePrimitives(const HydroState<2> &          u,
                  HydroState<2> &                q,
                  HydroSettings const &          settings,
                  EosWrapper_t<device_t> const & eos)
{
  const auto smallr = settings.smallr;

  // compute mixture (single fluid density)
  real_t d = fmax(u[FiveEq::ID0] + u[FiveEq::ID1], smallr);
  real_t ux = u[FiveEq::IU] / d;
  real_t uy = u[FiveEq::IV] / d;

  real_t ekin = HALF_F * (ux * ux + uy * uy);

  // Etot = Eint + Ekin
  real_t eint = u[FiveEq::IE] / d - ekin;

  // compute pressure and speed of sound
  const auto p = eos.mixture_pressure(
    d, eint, u[FiveEq::IPHI], 1 - u[FiveEq::IPHI], u[FiveEq::ID0], u[FiveEq::ID1]);

  q[FiveEq::ID] = d;
  q[FiveEq::ID0] = u[FiveEq::ID0];
  q[FiveEq::ID1] = u[FiveEq::ID1];
  q[FiveEq::IP] = p;
  q[FiveEq::IU] = ux;
  q[FiveEq::IV] = uy;
  q[FiveEq::IPHI] = sanitize_phi(u[FiveEq::IPHI]);

  hydro_sanitizer<2>(q);

  return eos.mixture_sound_speed(
    d, p, q[FiveEq::IPHI], 1 - q[FiveEq::IPHI], u[FiveEq::ID0], u[FiveEq::ID1]);

} // computePrimitives - 2d

// ====================================================================
// ====================================================================
/**
 * Convert conservative variables (phi0rho0, phi1rho1, phi , rho*u, rho*v, rho*w, e) to
 * primitive variables (rho,u,v,w,p)
 * \param[in]  u  conservative variables array (bifluid state)
 * \param[out] q  primitive    variables array (single fluid equivalent)
 *
 * \note for simplicity we use the same class HydroState for both conservative and primitive
 * variables even though primitive variable state is smaller.
 *
 * \return c local mixture speed of sound
 */
template <typename device_t>
KOKKOS_INLINE_FUNCTION real_t
computePrimitives(const HydroState<3> &          u,
                  HydroState<3> &                q,
                  HydroSettings const &          settings,
                  EosWrapper_t<device_t> const & eos)
{
  const auto smallr = settings.smallr;

  // compute mixture (single fluid density)
  real_t d = fmax(u[FiveEq::ID0] + u[FiveEq::ID1], smallr);
  real_t ux = u[FiveEq::IU] / d;
  real_t uy = u[FiveEq::IV] / d;
  real_t uz = u[FiveEq::IW] / d;

  real_t ekin = HALF_F * (ux * ux + uy * uy + uz * uz);

  // Etot = Eint + Ekin
  real_t eint = u[FiveEq::IE] / d - ekin;

  // compute pressure and speed of sound
  real_t p = eos.mixture_pressure(
    d, eint, u[FiveEq::IPHI], 1 - u[FiveEq::IPHI], u[FiveEq::ID0], u[FiveEq::ID1]);

  q[FiveEq::ID] = d;
  q[FiveEq::ID0] = u[FiveEq::ID0];
  q[FiveEq::ID1] = u[FiveEq::ID1];
  q[FiveEq::IP] = p;
  q[FiveEq::IU] = ux;
  q[FiveEq::IV] = uy;
  q[FiveEq::IW] = uz;
  q[FiveEq::IPHI] = sanitize_phi(u[FiveEq::IPHI]);

  hydro_sanitizer<3>(q);

  return eos.mixture_sound_speed(
    d, p, q[FiveEq::IPHI], 1 - q[FiveEq::IPHI], u[FiveEq::ID0], u[FiveEq::ID1]);

} // computePrimitives - 3d

// ====================================================================
// ====================================================================
/**
 * Compute mixture specific kinetic energy.
 *
 * \param[in] u is an hydro state vector (conservative variables)
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION real_t
compute_mixture_ekin(const HydroState<dim> & u, HydroSettings const & settings)
{
  const auto smallr = settings.smallr;

  // compute mixture (single fluid density)
  real_t d = fmax(u[FiveEq::ID0] + u[FiveEq::ID1], smallr);

  if constexpr (dim == 2)
  {

    const auto ux = u[FiveEq::IU] / d;
    const auto uy = u[FiveEq::IV] / d;

    return HALF_F * (ux * ux + uy * uy);
  }
  else if constexpr (dim == 3)
  {
    const auto ux = u[FiveEq::IU] / d;
    const auto uy = u[FiveEq::IV] / d;
    const auto uz = u[FiveEq::IW] / d;

    return HALF_F * (ux * ux + uy * uy + uz * uz);
  }

  return ZERO_F;
} // compute_mixture_ekin

// ====================================================================
// ====================================================================
/**
 * Compute mixture pressure.
 *
 * \param[in] u is an hydro state vector (conservative variables)
 */
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION real_t
compute_mixture_pressure(const HydroState<dim> &        u,
                         HydroSettings const &          settings,
                         EosWrapper_t<device_t> const & eos)
{
  const auto smallr = settings.smallr;

  // compute mixture (single fluid density)
  real_t d = fmax(u[FiveEq::ID0] + u[FiveEq::ID1], smallr);

  // compute specific kinetic energy
  const auto ekin = compute_mixture_ekin<dim>(u, settings);

  // compute internal energy using definition : Etot = Eint + Ekin
  const auto eint = u[FiveEq::IE] / d - ekin;

  // compute pressure
  return eos.mixture_pressure(
    d, eint, u[FiveEq::IPHI], 1 - u[FiveEq::IPHI], u[FiveEq::ID0], u[FiveEq::ID1]);

} // compute_mixture_pressure

} // namespace models

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_MODELS_UTILS_H_
