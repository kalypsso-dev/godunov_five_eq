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
#include <godunov_five_eq/models/HydroSettings.h>
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
  if (q[FiveEq<dim>::IAD0] < eps)
  {
    q[FiveEq<dim>::IAD0] = ZERO_F;
  }

  if (q[FiveEq<dim>::IAD1] < eps)
  {
    q[FiveEq<dim>::IAD1] = ZERO_F;
  }

  if (q[FiveEq<dim>::IA0] < eps)
  {
    q[FiveEq<dim>::IA0] = ZERO_F;
  }

  if (q[FiveEq<dim>::IA0] > ONE_F - eps)
  {
    q[FiveEq<dim>::IA0] = ONE_F;
  }

  if (q[FiveEq<dim>::IA1] < eps)
  {
    q[FiveEq<dim>::IA1] = ZERO_F;
  }

  if (q[FiveEq<dim>::IA1] > ONE_F - eps)
  {
    q[FiveEq<dim>::IA1] = ONE_F;
  }

} // hydro_sanitizer

// ====================================================================
// ====================================================================
/**
 * Ensure volume fraction is in range [0, 1]
 *
 * \param[in] alpha is volumic fraction
 */
KOKKOS_INLINE_FUNCTION
auto
sanitize_alpha(real_t alpha)
{
  return fmax(ZERO_F, fmin(ONE_F, alpha));
}

// ====================================================================
// ====================================================================
/**
 *
 */
KOKKOS_INLINE_FUNCTION
real_t
sanitize_dalpha(real_t alpha, real_t dalpha)
{
  const auto alpha_new = fmax(KALYPSSO_NUM(0.0), fmin(KALYPSSO_NUM(1.0), alpha + dalpha));
  return alpha_new - alpha;
}

// ====================================================================
// ====================================================================
template <size_t dim>
KOKKOS_INLINE_FUNCTION bool
is_valid_primitive_state(HydroState<dim> const & q)
{
  const auto   rho = q[FiveEq<dim>::IAD0] + q[FiveEq<dim>::IAD1];
  auto const & P = q[FiveEq<dim>::IP];
  auto const & alpha = q[FiveEq<dim>::IA0];

  return (q[FiveEq<dim>::IAD0] >= 0 and q[FiveEq<dim>::IAD1] >= 0 and P >= 0 and alpha >= ZERO_F and
          alpha <= ONE_F);

} // is_valid_primitive_state

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

  // get mixture density
  const auto d = fmax(u[FiveEq<dim>::ID], smallr);

  if constexpr (dim == 2)
  {
    const auto ux = u[FiveEq<dim>::IU] / d;
    const auto uy = u[FiveEq<dim>::IV] / d;

    return HALF_F * (ux * ux + uy * uy);
  }
  else if constexpr (dim == 3)
  {
    const auto ux = u[FiveEq<dim>::IU] / d;
    const auto uy = u[FiveEq<dim>::IV] / d;
    const auto uz = u[FiveEq<dim>::IW] / d;

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
  real_t d = fmax(u[FiveEq<dim>::IAD0] + u[FiveEq<dim>::IAD1], smallr);

  // compute specific kinetic energy
  const auto ekin = compute_mixture_ekin<dim>(u, settings);

  // compute internal energy using definition : Etot = Eint + Ekin
  const auto eint = u[FiveEq<dim>::IE] / d - ekin;

  // compute pressure
  return eos.mixture_pressure(d,
                              eint,
                              u[FiveEq<dim>::IA0],
                              1 - u[FiveEq<dim>::IA0],
                              u[FiveEq<dim>::IAD0],
                              u[FiveEq<dim>::IAD1]);

} // compute_mixture_pressure

// ====================================================================
// ====================================================================
/**
 * Convert conservative variables (alpha0rho0, alpha1rho1, alpha , rho*u, rho*v, rho*w, e) to
 * primitive variables (rho,u,v,w,p)
 * \param[in]  u  conservative variables array (bifluid state)
 * \param[out] q  primitive    variables array (single fluid equivalent)
 *
 * \note for simplicity we use the same class HydroState for both conservative and primitive
 * variables even though primitive variable state is smaller.
 *
 * \return c local mixture speed of sound
 */
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION real_t
computePrimitives(const HydroState<dim> &        u,
                  HydroState<dim> &              q,
                  HydroSettings const &          settings,
                  EosWrapper_t<device_t> const & eos)
{
  const auto smallr = settings.smallr;

  // compute mixture (single fluid density)
  const auto d = fmax(u[FiveEq<dim>::ID], smallr);
  const auto ux = u[FiveEq<dim>::IU] / d;
  const auto uy = u[FiveEq<dim>::IV] / d;

  const auto ekin = compute_mixture_ekin<dim>(u, settings);

  // Etot = Eint + Ekin
  const auto eint = u[FiveEq<dim>::IE] / d - ekin;

  // compute pressure and speed of sound
  const auto p = eos.mixture_pressure(d,
                                      eint,
                                      u[FiveEq<dim>::IA0],
                                      ONE_F - u[FiveEq<dim>::IA0],
                                      u[FiveEq<dim>::IAD0],
                                      u[FiveEq<dim>::IAD1]);

  q[FiveEq<dim>::ID] = d;
  q[FiveEq<dim>::IP] = p;
  q[FiveEq<dim>::IU] = ux;
  q[FiveEq<dim>::IV] = uy;
  if constexpr (dim == 3)
  {
    const auto uz = u[FiveEq<dim>::IW] / d;
    q[FiveEq<dim>::IW] = uz;
  }
  q[FiveEq<dim>::IA0] = sanitize_alpha(u[FiveEq<dim>::IA0]);
  q[FiveEq<dim>::IAD0] = u[FiveEq<dim>::IAD0];
  q[FiveEq<dim>::IA1] = ONE_F - q[FiveEq<dim>::IA0];
  q[FiveEq<dim>::IAD1] = u[FiveEq<dim>::IAD1];

  hydro_sanitizer<dim>(q);

  return eos.mixture_sound_speed(d,
                                 p,
                                 q[FiveEq<dim>::IA0],
                                 ONE_F - q[FiveEq<dim>::IA0],
                                 u[FiveEq<dim>::IAD0],
                                 u[FiveEq<dim>::IAD1]);

} // computePrimitives

} // namespace models

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_MODELS_UTILS_H_
