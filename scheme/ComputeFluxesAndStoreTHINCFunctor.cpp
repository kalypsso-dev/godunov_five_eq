// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeFluxesAndStoreTHINCFunctor.cpp
 */
#include <godunov_five_eq/scheme/ComputeFluxesAndStoreTHINCFunctor.h>

#include <kalypsso/core/mesh_utils.h>
#include <kalypsso/core/vof/youngs.h>

namespace kalypsso
{

namespace godunov_five_eq
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
ComputeFluxesAndStoreTHINCFunctor<dim, device_t>::ComputeFluxesAndStoreTHINCFunctor(
  orchard_key_view_t const &      orchard_keys,
  AMRMeshInfo const &             amr_mesh_info,
  DataArrayBlock_t const &        fluxes,
  DataArrayBlock_t const &        u_star,
  DataArrayGhostedBlock_t const & q_ghosted,
  DataArrayGhostedBlock_t const & slopes_x,
  DataArrayGhostedBlock_t const & slopes_y,
  DataArrayGhostedBlock_t const & slopes_z,
  int32_t                         iOct_flux_offset,
  int32_t                         num_quads,
  int                             direction,
  HydroSettings const &           hydro_settings,
  EosWrapper_t<device_t> const &  eos,
  real_t                          dt,
  real_t                          scaling_factor,
  TimeIntegrator                  time_integrator,
  THINCParams const &             thinc_params)
  : m_orchard_keys_device(orchard_keys)
  , m_amr_mesh_info(amr_mesh_info)
  , m_Fluxes(fluxes)
  , m_u_star(u_star)
  , m_q(q_ghosted)
  , m_slopes_x(slopes_x)
  , m_slopes_y(slopes_y)
  , m_slopes_z(slopes_z)
  , m_iOct_flux_offset(iOct_flux_offset)
  , m_num_quads(num_quads)
  , m_direction(direction)
  , m_block_sizes(slopes_x.block_size())
  , m_nbCellsPerLeaf(Kokkos::dim_prod(m_block_sizes))
  , m_hydro_settings(hydro_settings)
  , m_eos(eos)
  , m_dt(dt)
  , m_scaling_factor(scaling_factor)
  , m_time_integrator(time_integrator)
  , m_thinc(thinc_params)
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeFluxesAndStoreTHINCFunctor<dim, device_t>::apply(ConfigMap const &          config_map,
                                                        orchard_key_view_t const & orchard_keys,
                                                        AMRMeshInfo const &        amr_mesh_info,
                                                        DataArrayBlock_t const &   fluxes,
                                                        DataArrayBlock_t const &   u_star,
                                                        DataArrayGhostedBlock_t const & q_ghosted,
                                                        DataArrayGhostedBlock_t const & slopes_x,
                                                        DataArrayGhostedBlock_t const & slopes_y,
                                                        DataArrayGhostedBlock_t const & slopes_z,
                                                        int32_t               iOct_flux_offset,
                                                        int32_t               num_quads,
                                                        int                   direction,
                                                        HydroSettings const & hydro_settings,
                                                        EosWrapper_t<device_t> const & eos,
                                                        real_t                         dt)
{
  // Important note: the caller is responsible for providing a flux array with the right shape.
  {
    [[maybe_unused]] auto flux_block_sizes = q_ghosted.block_size();
    flux_block_sizes[direction]++;
    assertm(flux_block_sizes == fluxes.shape(), "Flux array has incompatible shape.");
  }

  ComputeFluxesAndStoreTHINCFunctor<dim, device_t> functor(
    orchard_keys,
    amr_mesh_info,
    fluxes,
    u_star,
    q_ghosted,
    slopes_x,
    slopes_y,
    slopes_z,
    iOct_flux_offset,
    num_quads,
    direction,
    hydro_settings,
    eos,
    dt,
    get_scaling_factor(config_map),
    TimeIntegratorConfig::get_time_integrator(config_map),
    THINCParams(config_map));

  const auto nbIterations = num_quads * fluxes.num_cells();

  // launch computation
  Kokkos::parallel_for("kalypsso::godunov_five_eq::ComputeFluxesAndStoreTHINCFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbIterations),
                       functor);

} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeFluxesAndStoreTHINCFunctor<dim, device_t>::reconstruct_state_2d(
  const HydroState<2> &   q,
  int32_t                 is,
  int32_t                 js,
  int32_t                 iOct_local,
  const offsets_t<2> &    offsets,
  [[maybe_unused]] real_t dtdx,
  [[maybe_unused]] real_t dtdy) const
{
  // auto const &                  smallr = m_hydro_settings.smallr;
  [[maybe_unused]] auto const & smallp = m_hydro_settings.smallp;

  // reconstruct state on interface
  HydroState<2> qr;

  const auto dir = offsets[IX] != 0 ? IX : IY;
  const auto uv = get_unit_vector<dim>(dir);

  // check monotonicity constraint
  // clang-format off
  const auto monotonicity =
    (m_q(is + uv[IX], js + uv[IY], Hydro::IA0, iOct_local) -
     m_q(is         , js         , Hydro::IA0, iOct_local)) *
    (m_q(is         , js         , Hydro::IA0, iOct_local) -
     m_q(is - uv[IX], js - uv[IY], Hydro::IA0, iOct_local));
  // clang-format on

  auto const & r0 = q[Hydro::IAD0];
  auto const & r1 = q[Hydro::IAD1];

  // THINC reconstruction is applied only in mixed cells when volume fraction is evolving
  // monotonically
  const bool thinc_requested = (q[Hydro::IA0] > m_thinc.epsilon) and
                               (q[Hydro::IA0] < ONE_F - m_thinc.epsilon) and monotonicity > 0;

  auto const drx0 = m_slopes_x(is, js, Hydro::IAD0, iOct_local);
  auto const drx1 = m_slopes_x(is, js, Hydro::IAD1, iOct_local);
  auto const dry0 = m_slopes_y(is, js, Hydro::IAD0, iOct_local);
  auto const dry1 = m_slopes_y(is, js, Hydro::IAD1, iOct_local);

  // reconstruction of volume fraction and phasic densities
  // we use notations from Zhang et al, JCP 498 (2024) 112672
  // equation (43)
  // and also from Zhang et al, JCP 524 (2025) 113702
  // equation (43)
  if (thinc_requested)
  {
    const auto       cell_size = init_kokkos_array<real_t, dim>(1);
    const coord_t<2> ijs{ is, js };

    // const auto normal_vector = vof::youngs_normal(m_q, ijs, Hydro::IA0], iOct_local,
    // cell_size);
    const auto normal_vector =
      vof::youngs_normal_inplace(m_q, ijs, Hydro::IA0, iOct_local, cell_size);

    // is it a left or right interface ?
    //  i-1  L|R     i    L|R    i+1
    // const int  a = offsets[dir] < 0 ? 1 : 0;
    const int  b = offsets[dir] < 0 ? -1 : 1;
    const auto beta = m_thinc.beta * fabs(normal_vector[dir]) + KALYPSSO_NUM(0.01);
    // const auto A = exp(2 * beta);
    // const auto B = exp(2 * beta * q[Hydro::IA0]);
    // const auto c = KALYPSSO_NUM(1.0) / 2 / beta * log((B - KALYPSSO_NUM(1.0)) / (A - B));
    // const auto sigma = COPYSIGN(ONE_F,
    //                             m_q(is + uv[IX], js + uv[IY], Hydro::IA0, iOct_local) -
    //                               m_q(is - uv[IX], js - uv[IY], Hydro::IA0, iOct_local));
    // const auto sigma_p1o2 = (sigma + ONE_F) / 2;

    const auto A = (m_q(is + uv[IX], js + uv[IY], Hydro::IA0, iOct_local) +
                    m_q(is - uv[IX], js - uv[IY], Hydro::IA0, iOct_local)) /
                   2;
    const auto B = (m_q(is + uv[IX], js + uv[IY], Hydro::IA0, iOct_local) -
                    m_q(is - uv[IX], js - uv[IY], Hydro::IA0, iOct_local)) /
                   2;
    const auto C = tanh(HALF_F * beta);
    const auto phi = m_q(is, js, Hydro::IA0, iOct_local);
    const auto D = tanh(HALF_F * beta * (phi - A) / B);

    // eq (41) - Zhang JCP 2025
    // qr[Hydro::IA0] =
    //   HALF_F *
    //   (ONE_F + tanh(beta * (static_cast<real_t>(a) + static_cast<real_t>(b) * sigma_p1o2 + c)));

    // eq (43) - Zhang JCP 2025
    qr[Hydro::IA0] = A + b * B * (C + b * D / C) / (ONE_F + b * D);

    // eq (45)
    // clang-format off
    qr[Hydro::IAD0] = r0 + r0 /          q[Hydro::IA0]  * (qr[Hydro::IA0] -  q[Hydro::IA0]);
    qr[Hydro::IAD1] = r1 + r1 / (ONE_F - q[Hydro::IA0]) * ( q[Hydro::IA0] - qr[Hydro::IA0]);
    // clang-format on
  }
  else
  {
    // use regular MUSCL reconstruction for volume fraction related quantity

    // just copy volumic fraction
    qr[Hydro::IA0] = q[Hydro::IA0];
    qr[Hydro::IAD0] = r0 + HALF_F * offsets[IX] * drx0 + HALF_F * offsets[IY] * dry0;
    qr[Hydro::IAD1] = r1 + HALF_F * offsets[IX] * drx1 + HALF_F * offsets[IY] * dry1;
  }

  //
  // then reconstruct other variables (pressure and velocities) using regular MUSCL
  //

  // retrieve primitive variables in current quadrant
  auto const & p = q[Hydro::IP];
  auto const & u = q[Hydro::IU];
  auto const & v = q[Hydro::IV];
  // auto const w = 0.0;
  const auto   r = r0 + r1;
  const real_t c = m_eos.mixture_sound_speed(r, p, q[Hydro::IA0], 1 - q[Hydro::IA0], r0, r1);

  auto const dpx = m_slopes_x(is, js, Hydro::IP, iOct_local);
  auto const dux = m_slopes_x(is, js, Hydro::IU, iOct_local);
  auto const dvx = m_slopes_x(is, js, Hydro::IV, iOct_local);
  // auto const dwx = 0.0;

  auto const dpy = m_slopes_y(is, js, Hydro::IP, iOct_local);
  auto const duy = m_slopes_y(is, js, Hydro::IU, iOct_local);
  auto const dvy = m_slopes_y(is, js, Hydro::IV, iOct_local);
  // auto const dwy = 0.0;

  qr[Hydro::IP] = p + HALF_F * offsets[IX] * dpx + HALF_F * offsets[IY] * dpy;
  qr[Hydro::IU] = u + HALF_F * offsets[IX] * dux + HALF_F * offsets[IY] * duy;
  qr[Hydro::IV] = v + HALF_F * offsets[IX] * dvx + HALF_F * offsets[IY] * dvy;

  if (m_time_integrator == +TimeIntegrator::HANCOCK)
  {
    // Add Hancock term: half time-step integration of non-conservative equations

    // clang-format off
    auto const sr0   = (-u * (drx0+drx1) - dux * r) * dtdx +
                       (-v * (dry0+dry1) - dvy * r) * dtdy;

    auto const sr0_0 = sr0 * q[Hydro::IA0];

    auto const sr0_1 = sr0 * (ONE_F - q[Hydro::IA0]);

    auto const su0 =   (-u * dux  - dpx / r ) * dtdx +
                       (-v * duy            ) * dtdy;
    auto const sv0 =   (-u * dvx            ) * dtdx +
                       (-v * dvy  - dpy / r ) * dtdy;
    auto const sp0 =   (-u * dpx  - dux * r * c * c) * dtdx +
                       (-v * dpy  - dvy * r * c * c) * dtdy;
    // clang-format on

    qr[Hydro::IAD0] += HALF_F * sr0_0;
    qr[Hydro::IAD1] += HALF_F * sr0_1;
    qr[Hydro::IP] += HALF_F * sp0;
    qr[Hydro::IU] += HALF_F * su0;
    qr[Hydro::IV] += HALF_F * sv0;
  }

  // qr[Hydro::IAD0] = fmax(smallr, qr[Hydro::IAD0]);
  // qr[Hydro::IAD1] = fmax(smallr, qr[Hydro::IAD1]);
  //  qr[Hydro::IP] = fmax(smallp * (qr[Hydro::IAD0] + qr[Hydro::IAD1]), qr[Hydro::IP]);

  return qr;

} // reconstruct_state_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeFluxesAndStoreTHINCFunctor<dim, device_t>::reconstruct_state_3d(
  const HydroState<3> &   q,
  int32_t                 is,
  int32_t                 js,
  int32_t                 ks,
  int32_t                 iOct_local,
  const offsets_t<3> &    offsets,
  [[maybe_unused]] real_t dtdx,
  [[maybe_unused]] real_t dtdy,
  [[maybe_unused]] real_t dtdz) const
{
  // auto const &                  smallr = m_hydro_settings.smallr;
  [[maybe_unused]] auto const & smallp = m_hydro_settings.smallp;

  // reconstruct state on interface
  HydroState<3> qr;

  const auto dir = offsets[IX] != 0 ? IX : offsets[IY] != 0 ? IY : IZ;
  const auto uv = get_unit_vector<dim>(dir);

  // check monotonicity constraint
  // clang-format off
  const auto monotonicity =
      (m_q(is + uv[IX], js + uv[IY], ks + uv[IZ], Hydro::IA0, iOct_local) -
       m_q(is         , js         , ks         , Hydro::IA0, iOct_local)) *
      (m_q(is         , js         , ks         , Hydro::IA0, iOct_local) -
       m_q(is - uv[IX], js - uv[IY], ks - uv[IZ], Hydro::IA0, iOct_local));
  // clang-format on

  // retrieve primitive variables in current quadrant
  const auto & r0 = q[Hydro::IAD0];
  const auto & r1 = q[Hydro::IAD1];

  auto const drx0 = m_slopes_x(is, js, ks, Hydro::IAD0, iOct_local);
  auto const drx1 = m_slopes_x(is, js, ks, Hydro::IAD1, iOct_local);
  auto const dry0 = m_slopes_y(is, js, ks, Hydro::IAD0, iOct_local);
  auto const dry1 = m_slopes_y(is, js, ks, Hydro::IAD1, iOct_local);
  auto const drz0 = m_slopes_z(is, js, ks, Hydro::IAD0, iOct_local);
  auto const drz1 = m_slopes_z(is, js, ks, Hydro::IAD1, iOct_local);

  // THINC reconstruction is applied only in mixed cells when volume fraction is evolving
  // monotonically
  const bool thinc_requested = (q[Hydro::IA0] > m_thinc.epsilon) and
                               (q[Hydro::IA0] < ONE_F - m_thinc.epsilon) and monotonicity > 0;

  // reconstruction of volume fraction and phasic densities
  // we use notations from Zhang et al, JCP 498 (2024) 112672
  // equation (43)
  if (thinc_requested)
  {
    const auto       cell_size = init_kokkos_array<real_t, dim>(1);
    const coord_t<3> ijs{ is, js, ks };

    // const auto normal_vector = vof::youngs_normal(m_q, ijs, Hydro::IA0], iOct_local,
    // cell_size);
    const auto normal_vector =
      vof::youngs_normal_inplace(m_q, ijs, Hydro::IA0, iOct_local, cell_size);

    // is it a left or right interface ?
    //  i-1  L|R     i    L|R    i+1
    // const int  a = offsets[dir] < 0 ? 1 : 0;
    const int  b = offsets[dir] < 0 ? -1 : 1;
    const auto beta = m_thinc.beta * fabs(normal_vector[dir]) + KALYPSSO_NUM(0.01);
    // const auto A = exp(2 * beta);
    // const auto B = exp(2 * beta * q[Hydro::IA0]);
    // const auto c = KALYPSSO_NUM(1.0) / 2 / beta * log((B - KALYPSSO_NUM(1.0)) / (A - B));
    // const auto sigma =
    //   COPYSIGN(ONE_F,
    //            m_q(is + uv[IX], js + uv[IY], ks + uv[IZ], Hydro::IA0, iOct_local) -
    //              m_q(is - uv[IX], js - uv[IY], ks - uv[IZ], Hydro::IA0, iOct_local));
    // const auto sigma_p1o2 = (sigma + ONE_F) / 2;

    const auto A = (m_q(is + uv[IX], js + uv[IY], ks + uv[IZ], Hydro::IA0, iOct_local) +
                    m_q(is - uv[IX], js - uv[IY], ks - uv[IZ], Hydro::IA0, iOct_local)) /
                   2;
    const auto B = (m_q(is + uv[IX], js + uv[IY], ks + uv[IZ], Hydro::IA0, iOct_local) -
                    m_q(is - uv[IX], js - uv[IY], ks - uv[IZ], Hydro::IA0, iOct_local)) /
                   2;
    const auto C = tanh(HALF_F * beta);
    const auto phi = m_q(is, js, ks, Hydro::IA0, iOct_local);
    const auto D = tanh(HALF_F * beta * (phi - A) / B);

    // eq (41) - Zhang JCP 2025
    // qr[Hydro::IA0] =
    //   HALF_F *
    //   (ONE_F + tanh(beta * (static_cast<real_t>(a) + static_cast<real_t>(b) * sigma_p1o2 + c)));

    // eq (43) - Zhang JCP 2025
    qr[Hydro::IA0] = A + b * B * (C + b * D / C) / (ONE_F + b * D);

    // eq (45)
    // clang-format off
    qr[Hydro::IAD0] = r0 + r0 /          q[Hydro::IA0]  * (qr[Hydro::IA0] -  q[Hydro::IA0]);
    qr[Hydro::IAD1] = r1 + r1 / (ONE_F - q[Hydro::IA0]) * ( q[Hydro::IA0] - qr[Hydro::IA0]);
    // clang-format on
  }
  else
  {
    // use regular MUSCL reconstruction for volume fraction related quantity
    // just copy volumic fraction
    qr[Hydro::IA0] = q[Hydro::IA0];
    qr[Hydro::IAD0] =
      r0 + HALF_F * offsets[IX] * drx0 + HALF_F * offsets[IY] * dry0 + HALF_F * offsets[IZ] * drz0;
    qr[Hydro::IAD1] =
      r1 + HALF_F * offsets[IX] * drx1 + HALF_F * offsets[IY] * dry1 + HALF_F * offsets[IZ] * drz1;
  }

  //
  // then reconstruct other variables (pressure and velocities) using regular MUSCL
  //

  const auto & p = q[Hydro::IP];
  const auto & u = q[Hydro::IU];
  const auto & v = q[Hydro::IV];
  const auto & w = q[Hydro::IW];
  const auto   r = r0 + r1;
  const real_t c = m_eos.mixture_sound_speed(r, p, q[Hydro::IA0], 1 - q[Hydro::IA0], r0, r1);

  // retrieve variations = dx * slopes
  const auto dpx = m_slopes_x(is, js, ks, Hydro::IP, iOct_local);
  const auto dux = m_slopes_x(is, js, ks, Hydro::IU, iOct_local);
  const auto dvx = m_slopes_x(is, js, ks, Hydro::IV, iOct_local);
  const auto dwx = m_slopes_x(is, js, ks, Hydro::IW, iOct_local);

  const auto dpy = m_slopes_y(is, js, ks, Hydro::IP, iOct_local);
  const auto duy = m_slopes_y(is, js, ks, Hydro::IU, iOct_local);
  const auto dvy = m_slopes_y(is, js, ks, Hydro::IV, iOct_local);
  const auto dwy = m_slopes_y(is, js, ks, Hydro::IW, iOct_local);

  const auto dpz = m_slopes_z(is, js, ks, Hydro::IP, iOct_local);
  const auto duz = m_slopes_z(is, js, ks, Hydro::IU, iOct_local);
  const auto dvz = m_slopes_z(is, js, ks, Hydro::IV, iOct_local);
  const auto dwz = m_slopes_z(is, js, ks, Hydro::IW, iOct_local);

  qr[Hydro::IP] =
    p + HALF_F * offsets[IX] * dpx + HALF_F * offsets[IY] * dpy + HALF_F * offsets[IZ] * dpz;
  qr[Hydro::IU] =
    u + HALF_F * offsets[IX] * dux + HALF_F * offsets[IY] * duy + HALF_F * offsets[IZ] * duz;
  qr[Hydro::IV] =
    v + HALF_F * offsets[IX] * dvx + HALF_F * offsets[IY] * dvy + HALF_F * offsets[IZ] * dvz;
  qr[Hydro::IW] =
    w + HALF_F * offsets[IX] * dwx + HALF_F * offsets[IY] * dwy + HALF_F * offsets[IZ] * dwz;

  if (m_time_integrator == +TimeIntegrator::HANCOCK)
  {
    // MUSCL-Hancock half time-step
    // clang-format off
    const auto sr0   = (-u * (drx0+drx1) - dux * r) * dtdx +
                       (-v * (dry0+dry1) - dvy * r) * dtdy +
                       (-w * (drz0+drz1) - dwz * r) * dtdz;

    auto const sr0_0 = sr0 * q[Hydro::IA0];

    auto const sr0_1 = sr0 * (ONE_F - q[Hydro::IA0]);

    const auto su0 =   (-u * dux - dpx / r)   * dtdx +
                       (-v * duy          )   * dtdy +
                       (-w * duz          )   * dtdz;
    const auto sv0 =   (-u * dvx          )   * dtdx +
                       (-v * dvy - dpy / r)   * dtdy +
                       (-w * dvz          )   * dtdz;
    const auto sw0 =   (-u * dwx          )   * dtdx +
                       (-v * dwy          )   * dtdy +
                       (-w * dwz - dpz / r)   * dtdz;
    const auto sp0 =   (-u * dpx - dux * r * c * c) * dtdx +
                       (-v * dpy - dvy * r * c * c) * dtdy +
                       (-w * dpz - dwz * r * c * c) * dtdz;
    // clang-format on

    qr[Hydro::IAD0] += HALF_F * sr0_0;
    qr[Hydro::IAD1] += HALF_F * sr0_1;
    qr[Hydro::IP] += HALF_F * sp0;
    qr[Hydro::IU] += HALF_F * su0;
    qr[Hydro::IV] += HALF_F * sv0;
    qr[Hydro::IW] += HALF_F * sw0;
  }

  // qr[Hydro::IAD0] = fmax(smallr, qr[Hydro::IAD0]);
  // qr[Hydro::IAD1] = fmax(smallr, qr[Hydro::IAD1]);
  //  qr[Hydro::IP] = fmax(smallp * (qr[Hydro::IAD0] + qr[Hydro::IAD1]), qr[Hydro::IP]);

  return qr;

} // reconstruct_state_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndStoreTHINCFunctor<dim, device_t>::compute_fluxes_and_store_2d(
  int32_t const & cell_index,
  int32_t const & iOct_local) const
{

  auto const coords = cell_index_unravel<2>(cell_index, m_Fluxes.shape());

  auto const & i = coords[IX];
  auto const & j = coords[IY];

  // index to address slope arrays
  auto const is = i;
  auto const js = j;

  // index to address primitive variables arrays
  auto const iq = i;
  auto const jq = j;

  // get AMR level
  auto const iOct_global = iOct_local + m_iOct_flux_offset;
  auto const level = orchard_key_t<2>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 4 times larger than current cell volume
  auto const dx = compute_cell_length<2>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  auto qprim = get_prim_variables(iq, jq, iOct_local);

  /*
   * compute flux from left face along X dir and update both sides
   */
  if (m_direction == IX)
  {
    // get state in neighbor along X
    auto qprim_n = get_prim_variables(iq - 1, jq, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets{ 1, 0 };

    // reconstruct state in left neighbor (index relative to slopes array)
    auto qL = reconstruct_state_2d(
      qprim_n, is - 1, js, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // step 2 : reconstruct state in current cell
    offsets = { -1, 0 };

    // reconstruct state from current cell center to left interface
    auto qR =
      reconstruct_state_2d(qprim, is, js, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // step 3 : compute flux (Riemann solver)
    auto              riemann_state = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);
    HydroState<dim> & flux = riemann_state.flux;
    auto &            ustar = riemann_state.ustar;
    auto &            phistar = riemann_state.phistar;

    flux[Hydro::IA0] = ustar * phistar;

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, iOct_local, flux_cur, ustar * dtdS_over_dV_cur);
  }

  /*
   * compute flux from left face along Y dir and update both sides
   */
  if (m_direction == IY)
  {
    // get state in neighbor along Y
    auto qprim_n = get_prim_variables(iq, jq - 1, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets = { 0, 1 };

    // reconstruct "left" state
    auto qL = reconstruct_state_2d(
      qprim_n, is, js - 1, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // step 2 : reconstruct state in current cell
    offsets = { 0, -1 };

    // reconstruct state from current cell center to left interface
    auto qR =
      reconstruct_state_2d(qprim, is, js, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // swap IU / IV
    my_swap(qL[Hydro::IU], qL[Hydro::IV]);
    my_swap(qR[Hydro::IU], qR[Hydro::IV]);

    // step 3 : compute flux (Riemann solver)
    auto              riemann_state = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);
    HydroState<dim> & flux = riemann_state.flux;
    auto &            ustar = riemann_state.ustar;
    auto &            phistar = riemann_state.phistar;

    my_swap(flux[Hydro::IU], flux[Hydro::IV]);

    flux[Hydro::IA0] = ustar * phistar;

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, iOct_local, flux_cur, ustar * dtdS_over_dV_cur);
  }

} // compute_fluxes_and_store_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndStoreTHINCFunctor<dim, device_t>::compute_fluxes_and_store_3d(
  const int32_t & cell_index,
  const int32_t & iOct_local) const
{
  auto const coords = cell_index_unravel<3>(cell_index, m_Fluxes.shape());

  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

  // index to address slope arrays
  auto const is = i;
  auto const js = j;
  auto const ks = k;

  // index to address primitive variables arrays
  auto const iq = i;
  auto const jq = j;
  auto const kq = k;

  // get AMR level
  auto const iOct_global = iOct_local + m_iOct_flux_offset;
  auto const level = orchard_key_t<3>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 8 times larger than current cell volume
  auto const dx = compute_cell_length<3>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  auto qprim = get_prim_variables(iq, jq, kq, iOct_local);

  /*
   * compute flux from left face along X dir
   */
  if (m_direction == IX)
  {
    // get state in neighbor along X
    auto qprim_n = get_prim_variables(iq - 1, jq, kq, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets{ 1, 0, 0 };

    // reconstruct state in left neighbor (index relative to slopes array)
    auto qL = reconstruct_state_3d(qprim_n,
                                   is - 1,
                                   js,
                                   ks,
                                   iOct_local,
                                   offsets,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur);

    // step 2 : reconstruct state in current cell
    offsets = { -1, 0, 0 };

    // reconstruct state from current cell center to left interface
    auto qR = reconstruct_state_3d(
      qprim, is, js, ks, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // step 3 : compute flux (Riemann solver)
    auto              riemann_state = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);
    HydroState<dim> & flux = riemann_state.flux;
    auto &            ustar = riemann_state.ustar;
    auto &            phistar = riemann_state.phistar;

    flux[Hydro::IA0] = ustar * phistar;

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur, ustar * dtdS_over_dV_cur);
  } // end update along X

  /*
   * compute flux from left face along Y dir
   */
  if (m_direction == IY)
  {
    // get state in neighbor along Y
    auto qprim_n = get_prim_variables(iq, jq - 1, kq, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets = { 0, 1, 0 };

    // reconstruct "left" state
    auto qL = reconstruct_state_3d(qprim_n,
                                   is,
                                   js - 1,
                                   ks,
                                   iOct_local,
                                   offsets,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur);

    offsets = { 0, -1, 0 };

    auto qR = reconstruct_state_3d(
      qprim, is, js, ks, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // swap IU / IV
    my_swap(qL[Hydro::IU], qL[Hydro::IV]);
    my_swap(qR[Hydro::IU], qR[Hydro::IV]);

    // step 3 : compute flux (Riemann solver)
    auto              riemann_state = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);
    HydroState<dim> & flux = riemann_state.flux;
    auto &            ustar = riemann_state.ustar;
    auto &            phistar = riemann_state.phistar;

    my_swap(flux[Hydro::IU], flux[Hydro::IV]);

    flux[Hydro::IA0] = ustar * phistar;

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur, ustar * dtdS_over_dV_cur);
  } // end update along Y

  /*
   * compute flux from left face along Z dir
   */
  if (m_direction == IZ)
  {
    // get state in neighbor along Z
    auto qprim_n = get_prim_variables(iq, jq, kq - 1, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets = { 0, 0, 1 };

    // reconstruct "left" state
    auto qL = reconstruct_state_3d(qprim_n,
                                   is,
                                   js,
                                   ks - 1,
                                   iOct_local,
                                   offsets,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur);

    offsets = { 0, 0, -1 };

    auto qR = reconstruct_state_3d(
      qprim, is, js, ks, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // swap IU / IW
    my_swap(qL[Hydro::IU], qL[Hydro::IW]);
    my_swap(qR[Hydro::IU], qR[Hydro::IW]);

    // step 3 : compute flux (Riemann solver)
    auto              riemann_state = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);
    HydroState<dim> & flux = riemann_state.flux;
    auto &            ustar = riemann_state.ustar;
    auto &            phistar = riemann_state.phistar;

    my_swap(flux[Hydro::IU], flux[Hydro::IW]);

    flux[Hydro::IA0] = ustar * phistar;

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur, ustar * dtdS_over_dV_cur);
  } // end update along Z

} // compute_fluxes_and_store_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndStoreTHINCFunctor<dim, device_t>::operator()(const index_t & global_index) const
{

  // retrieve local octant index in range [0, num_quads_to_process [
  auto const iOct_local = static_cast<int32_t>(global_index / m_Fluxes.num_cells());
  auto const cell_index = static_cast<int32_t>(global_index - iOct_local * m_Fluxes.num_cells());

  if constexpr (dim == 2)
  {
    compute_fluxes_and_store_2d(cell_index, iOct_local);
  }
  else if constexpr (dim == 3)
  {
    compute_fluxes_and_store_3d(cell_index, iOct_local);
  }

} // operator ()

// explicit template instantiation
template class ComputeFluxesAndStoreTHINCFunctor<2, kalypsso::DefaultDevice>;
template class ComputeFluxesAndStoreTHINCFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_five_eq

} // namespace kalypsso
