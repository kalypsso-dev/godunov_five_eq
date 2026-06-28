// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FiveEq.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_MODELS_FIVE_EQ_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_MODELS_FIVE_EQ_H_

#include <cstdint>
#include <string>

namespace kalypsso
{

namespace godunov_five_eq
{

namespace models
{

// =============================================================
// =============================================================
template <size_t dim>
class FiveEq
{

public:
  using Id_t = int32_t;

  /**
   * Five equations models field ids
   */
  enum VarId : Id_t
  {
    ID = 0,                        /*!< ID mixture density */
    IP = 1,                        /*!< IP Pressure/Energy field index */
    IE = 1,                        /*!< IE Energy/Pressure field index */
    IU = 2,                        /*!< X velocity / momentum index */
    IV = 3,                        /*!< Y velocity / momentum index */
    IW = (dim == 2) ? IV : IV + 1, /*!< Z velocity / momentum index */
    HYDRO_VARID_COUNT = IW + 1,    /*!< number of hydrodynamics variables */
    IA0 = HYDRO_VARID_COUNT,       /*!< IA0  volumic fraction           (material 0) index */
    IAD0 = HYDRO_VARID_COUNT + 1,  /*!< IAD0 volumic fraction x Density (material 0) index */
    IA1 = IA0 + 2,                 /*!< IA1  volumic fraction           (material 1) index */
    IAD1 = IAD0 + 2,               /*!< IAD1 volumic fraction x Density (material 1) index */
    VARID_COUNT = IAD1 + 1,        /*!< invalid index, just counting number of fields */
    INVALID_ID
  };

  enum MaterialId : Id_t
  {
    ALPHA = 0,
    ALPHA_RHO = 1,
    MATERIAL_ID_COUNT = 2
  };

  using var_names_t = std::array<std::string, static_cast<size_t>(VARID_COUNT)>;
  static const var_names_t ID_TO_NAMES;

  KOKKOS_INLINE_FUNCTION static int32_t
  IA(int32_t i_mat)
  {
    return IA0 + 2 * i_mat;
  }

  KOKKOS_INLINE_FUNCTION static int32_t
  IA(size_t i_mat)
  {
    return IA(static_cast<int32_t>(i_mat));
  }

  KOKKOS_INLINE_FUNCTION static int32_t
  IAD(int32_t i_mat)
  {
    return IAD0 + 2 * i_mat;
  }

  KOKKOS_INLINE_FUNCTION static int32_t
  IAD(size_t i_mat)
  {
    return IAD(static_cast<int32_t>(i_mat));
  }

  KOKKOS_INLINE_FUNCTION static constexpr size_t
  nbvar()
  {
    return static_cast<size_t>(HYDRO_VARID_COUNT + 2 * 2);
  }

  static std::string
  name(Id_t var)
  {
    if (var < HYDRO_VARID_COUNT)
      return ID_TO_NAMES[var];

    const auto  i_mat = (var - HYDRO_VARID_COUNT) / 2;
    const auto  i_var = (var - HYDRO_VARID_COUNT) % 2;
    std::string name = ID_TO_NAMES[IA0 + i_var];
    return name + '_' + std::to_string(i_mat);
  }

  static std::string
  name(size_t var)
  {
    return name(static_cast<Id_t>(var));
  }

  static int32_t
  id_from_name(std::string const & a_name)
  {
    for (int32_t i_var = 0; i_var < static_cast<int32_t>(FiveEq<dim>::nbvar()); ++i_var)
    {
      auto const varName = FiveEq<dim>::name(i_var);
      if (varName == a_name)
        return i_var;
    }
    return FiveEq<dim>::INVALID_ID;
  }

  static var_names_t
  get_var_names()
  {
    var_names_t arr;

    arr[ID] = "rho_mix";
    arr[IE] = "e_tot";
    arr[IU] = "rho_vx";
    arr[IV] = "rho_vy";
    if constexpr (dim == 3)
      arr[IW] = "rho_vz";
    arr[IA0] = "alpha";
    arr[IAD0] = "alpha_rho";

    return arr;
  }

}; // class FiveEq

template <size_t dim>
const typename FiveEq<dim>::var_names_t FiveEq<dim>::ID_TO_NAMES = FiveEq<dim>::get_var_names();

} // namespace models

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_MODELS_FIVE_EQ_H_
