// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeHancockUpdate.h
 *
 * Muscl-Hancock is a second order in time and space Godunov method.
 * Here we isolate the Hancock half time step in order to decouple space reconstruction from time
 * integration. This we allow to have either MUSCL-Hancock or THINC-Hancock half time step.
 *
 * Just as a reminder, here are the main steps of the MUSCL6hancock algorithm:
 *
 * 1. Compute primitive variables from conservative ones
 * (with the help of an equation of state): \f${\bf Q}_{i}^n({\bf U}_{i}^n)\f$
 *
 * 2. Space reconstruction of primitive variables from cell-center to left and right
 * face center at time \f$t_n\f$:
 * - \f${\bf Q}_{i,L}^n={\bf Q}_{i}^n-\frac{1}{2}\overline{\delta}_i\f$,
 * - \f${\bf Q}_{i,R}^n={\bf Q}_{i}^n+\frac{1}{2}\overline{\delta}_i\f$
 *
 * 3. Half time step evolution of face reconstructed states:
 *
 * \f{align*}{
   {\bf U}_{i,L}^{n+1/2}& ={\bf U}_{i,L}^n-\frac{\Delta t}{2\Delta x}\left[ {\bf F}_x({\bf
 Q}_{i,R}^n) -{\bf F}_x({\bf Q}_{i,L}^n)\right]\\
   {\bf U}_{i,R}^{n+1/2}& ={\bf U}_{i,R}^n-\frac{\Delta t}{2\Delta x}\left[ {\bf F}_x({\bf
 Q}_{i,R}^n) -{\bf F}_x({\bf Q}_{i,L}^n)\right]
  \f}
 *
 * 4. Compute flux using a Riemann solver (e.g. HLLC):
 *  \f${\bf F}_{x,i-1/2}^{n+1/2}=RS({\bf U}_{i-1,R}^{n+1/2},{\bf U}_{i,L}^{n+1/2})\f$
 *
 * 5.Time update from \f$t_n\f$ to \f$t_{n+1}\f$:
 *
 *  \f${\bf U}_{i}^{n+1}={\bf U}_{i}^{n}-\frac{\Delta}{\Delta x}\left[ {\bf
 F}_{x,i+1/2}^{n+1/2}-{\bf F}_{x,i-1/2}^{n+1/2}\right]\f$
 *
 *
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_COMPUTEHANCOCKUPDATE_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_COMPUTEHANCOCKUPDATE_H_

#include <godunov_five_eq/models/FiveEq.h>

namespace kalypsso
{

namespace godunov_five_eq
{

/**
 * Compute hydrodynamics fluxes.
 *
 * \param[in] q primitive variables
 *
 * \return hydrodynamics flux in direction dir
 */
template <size_t dim, int dir>
KOKKOS_FUNCTION HydroState<dim>
                compute_hydro_flux(const HydroState<dim> & q)
{}

} // namespace godunov_five_eq
} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_COMPUTEHANCOCKUPDATE_H_
