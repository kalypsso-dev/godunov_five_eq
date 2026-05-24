// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HydroState.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_MODELS_HYDRO_STATE_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_MODELS_HYDRO_STATE_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <godunov_five_eq/models/FiveEq.h>

namespace kalypsso
{

namespace godunov_five_eq
{

template <size_t nbvar>
using StateNd = Kokkos::Array<real_t, nbvar>;

/**
 * HydroState is a small array used for storing either primitive or conservative variables.
 *
 * \note the size is the number of independent degrees of freedom plus one for mixture density.
 */
template <size_t dim>
using HydroState = StateNd<models::FiveEq<dim>::nbvar()>;

using HydroState2d = HydroState<2>;
using HydroState3d = HydroState<3>;
using GravityState = Kokkos::Array<real_t, 3>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_MODELS_HYDRO_STATE_H_
