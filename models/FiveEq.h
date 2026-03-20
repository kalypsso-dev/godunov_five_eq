// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FiveEq.h
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_MODELS_FIVE_EQ_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_MODELS_FIVE_EQ_H_

#include <kalypsso/core/FieldMap.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace kalypsso
{

namespace godunov_five_eq
{

namespace models
{

// =============================================================
// =============================================================
class FiveEq
{

public:
  struct Settings
  {
    size_t dim = 2;

    Settings() = default;
    Settings(size_t dim_)
      : dim(dim_)
    {}
  }; // struct Settings

  using Id_t = int32_t;

  //! Five equations models field ids
  enum VarId : Id_t
  {
    ID0 = 0,    /*!< ID0 volumic fraction x Density (material 0) index */
    ID1 = 1,    /*!< ID1 volumic fraction x Density (material 1) index */
    ID = 2,     /*!< ID mixture density */
    IPHI = 3,   /*!< IPHI volumic fraction of material 0 */
    IP = 4,     /*!< IP Pressure/Energy field index */
    IE = 4,     /*!< IE Energy/Pressure field index */
    IU = 5,     /*!< X velocity / momentum index */
    IV = 6,     /*!< Y velocity / momentum index */
    IW = 7,     /*!< Z velocity / momentum index */
    IUSTAR = 8, /*!< IUSTAR star velocity used to compute volumic fraction advection source term */
    IGX = 9,    /*!< X gravitational field index */
    IGY = 10,   /*!< Y gravitational field index */
    IGZ = 11,   /*!< Z gravitational field index */
    VARID_COUNT = 12 /*!< invalid index, just counting number of fields */
  };

  //! a dictionary of all variables names and corresponding id (enum)
  //! this map is initialized in FiveEq.cpp
  static const id2names_t m_id2names_all;

public:
  FiveEq()
    : m_settings()
  {
    setup();
  }

  //! constructor
  FiveEq(Settings settings)
    : m_settings(settings)
  {
    setup();
  }

  //! constructor
  FiveEq(size_t dim)
    : m_settings(dim)
  {
    setup();
  }

  //! initialize model enabled variable maps
  void
  setup()
  {

    m_fieldmap.enable(ID0);
    m_fieldmap.enable(ID1);
    m_fieldmap.enable(IPHI);
    m_fieldmap.enable(IE);
    m_fieldmap.enable(IU);
    m_fieldmap.enable(IV);
    if (m_settings.dim == 3)
    {
      m_fieldmap.enable(IW);
    }
    m_fieldmap.enable(ID);
    m_fieldmap.enable(IUSTAR);

    // insert some fields in names2id and id2names maps
    for (int32_t varIdInt = 0; varIdInt != VARID_COUNT; ++varIdInt)
    {
      const VarId varId = static_cast<VarId>(varIdInt);
      if (m_fieldmap.enabled(varId))
      {
        const auto iter = m_id2names_all.find(varId);

        if (iter != m_id2names_all.end())
        {
          m_names2id[iter->second] = varId;
          m_id2names[varId] = iter->second;
        }
      }
    } // for varIdInt

  } // setup

  FieldMap<FiveEq>
  get_fieldmap()
  {
    return m_fieldmap;
  }

  const FieldMap<FiveEq> &
  get_fieldmap() const
  {
    return m_fieldmap;
  }

  //! get id to names map for enabled variables
  const names2id_t &
  get_names2id_map() const
  {
    return m_names2id;
  }

  //! get names to id map for enabled variables
  const id2names_t &
  get_id2names_map() const
  {
    return m_id2names;
  }

private:
  //! model settings
  Settings m_settings;

  //! Kokkos::Array mapping enums to index of active variables
  FieldMap<FiveEq> m_fieldmap;

  //! map of ids to names
  names2id_t m_names2id;

  //! map of ids to names
  id2names_t m_id2names;

}; // class FiveEq

} // namespace models

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_MODELS_FIVE_EQ_H_
