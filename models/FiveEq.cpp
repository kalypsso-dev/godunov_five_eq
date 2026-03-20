// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FiveEq.cpp
 */
#include <godunov_five_eq/models/FiveEq.h>

namespace kalypsso
{

namespace godunov_five_eq
{

namespace models
{

// clang-format off
const id2names_t FiveEq::m_id2names_all = {
  { FiveEq::ID0, "rho0" },
  { FiveEq::ID1, "rho1" },
  { FiveEq::ID, "rho_mix" },
  { FiveEq::IPHI, "phi" },
  { FiveEq::IE, "e_tot" },
  { FiveEq::IU, "rho_vx" },
  { FiveEq::IV, "rho_vy" },
  { FiveEq::IW, "rho_vz" },
  { FiveEq::IGX, "grav_x" },
  { FiveEq::IGY, "grav_y" },
  { FiveEq::IGZ, "grav_z" }
};
// clang-format on

} // namespace models

} // namespace godunov_five_eq

} // namespace kalypsso
