// SPDX-FileCopyrightText: 2026 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file eos_utils.h
 *
 * \brief Contains a shared set of values and types.
 */

#ifndef KALYPSSO_GODUNOV_FIVE_EQ_EOS_EOS_UTILS_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_EOS_EOS_UTILS_H_

#include <kalypsso/core/kalypsso_core_config.h>

#include <godunov_five_eq/eos/EosWrapper.h>
#include <kalypsso/core/eos/MieGruneisenMixture.h>

namespace kalypsso
{

namespace godunov_five_eq
{

template <typename device_t>
using EosWrapper_t = eos::EosWrapper<device_t>;

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_EOS_EOS_UTILS_H_
