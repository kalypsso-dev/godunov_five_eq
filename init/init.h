// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file init.h
 *
 * \brief Contains the function that initialises a solver.
 */

#ifndef KALYPSSO_GODUNOV_FIVE_EQ_INIT_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_INIT_H_

#include <godunov_five_eq/common.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// =======================================================
// =======================================================
/**
 * \brief Initializes conservative variables at time t=0 using analytical values.
 *
 * \param solver The solver itself.
 *
 */
template <size_t dim, typename device_t>
void
init(SolverGodunovFiveEq<dim, device_t> & solver);

// =======================================================
// =======================================================
/**
 * \brief Initializes the solver and its conservative values data using a previous run.
 *
 * The input parameter file must contain the of the file to read.
 */
template <size_t dim, typename device_t>
void
init_restart([[maybe_unused]] SolverGodunovFiveEq<dim, device_t> & solver);

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_INIT_H_
