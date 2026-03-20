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

namespace kalypsso
{

namespace godunov_five_eq
{

//! Number of conservative variables for the Five equations model in 2d
//! - one extra variable because 2 momentum equations (only 1 in 1d)
constexpr int FIVEEQ_2D_NBVAR = 5 + 1;

//! Number of conservative variables for the Five equations model in 3d
//! - two extra variables because 3 momentum equations (only 1 in 1d)
constexpr int FIVEEQ_3D_NBVAR = 5 + 2;

//! constant to be used as template parameter for nbvar_five_eq
constexpr bool DO_INCLUDE_MIXTURE_DENSITY = true;

//! constant to be used as template parameter for nbvar_five_eq
constexpr bool DONT_INCLUDE_MIXTURE_DENSITY = false;

/**
 * Number of variables used in conservative / primitive variables.
 *
 * \param include_mixture_density used to indicate if we want to store mixture density or not
 */
template <size_t dim, bool include_mixture_density = DONT_INCLUDE_MIXTURE_DENSITY>
constexpr auto
nbvar_five_eq()
{
  if constexpr (dim == 2)
    return include_mixture_density ? FIVEEQ_2D_NBVAR + 1 : FIVEEQ_2D_NBVAR;
  else if constexpr (dim == 3)
    return include_mixture_density ? FIVEEQ_3D_NBVAR + 1 : FIVEEQ_3D_NBVAR;
}

template <size_t nbvar>
using StateNd = Kokkos::Array<real_t, nbvar>;

/**
 * HydroState is a small array used for storing either primitive or conservative variables.
 *
 * \note the size is the number of independent degrees of freedom plus one for mixture density.
 */
template <size_t dim>
using HydroState = StateNd<nbvar_five_eq<dim, DO_INCLUDE_MIXTURE_DENSITY>()>;

using HydroState2d = HydroState<2>;
using HydroState3d = HydroState<3>;
using GravityState = Kokkos::Array<real_t, 3>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_MODELS_HYDRO_STATE_H_
