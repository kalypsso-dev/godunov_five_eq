// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file RiemannSolvers.h
 *
 * Some Riemann solver implementation for two fluid flow.
 */
#ifndef KALYPSSO_GODUNOV_FIVE_EQ_MODELS_RIEMANN_SOLVERS_H_
#define KALYPSSO_GODUNOV_FIVE_EQ_MODELS_RIEMANN_SOLVERS_H_

#include <math.h>

#include <kalypsso/core/HydroParams.h>
#include <kalypsso/core/real_type.h>

#include <kalypsso/core/models/riemann_solver_types.h>
#include <godunov_five_eq/models/HydroState.h>
#include <godunov_five_eq/models/FiveEq.h>
#include <godunov_five_eq/eos/eos_utils.h>

namespace kalypsso
{

namespace godunov_five_eq
{

template <size_t dim>
struct RiemannState
{
  HydroState<dim> flux;
  real_t          ustar;
  real_t          phistar;
};

// =====================================================================================
// =====================================================================================
/**
 * Riemann solver HLLC
 *
 * \param[in] qleft    : input left  state (primitive variables)
 * \param[in] qright   : input right state (primitive variables)
 * \param[out] flux    : output flux
 * \param[out] ustar   : velocity
 * \param[out] phistar : volumic fraction
 *
 *
 * Reference:
 * - http://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf
 * - "On the Choice of Wavespeeds for the HLLC Riemann Solver", Batten et al., SIAM J. Sci. Comp.,
 * vol 18, issue 6, 1997, https://doi.org/10.1137/S1064827593260140
 *
 */
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION RiemannState<dim>
                       riemann_hllc(HydroState<dim> const &        qleft,
                                    HydroState<dim> const &        qright,
                                    HydroSettings const &          settings,
                                    EosWrapper_t<device_t> const & eos)
{
  RiemannState<dim> riemann_state;
  auto &            flux = riemann_state.flux;
  auto &            ustar = riemann_state.ustar;
  auto &            phistar = riemann_state.phistar;

  // makes enum Hydro::VarId available
  using Hydro = kalypsso::godunov_five_eq::models::FiveEq;

  auto const & smallr = settings.smallr;
  auto const & smallp = settings.smallp;
  auto const & smallc = settings.smallc;

  //
  // Left variables
  //
  const auto rl = fmax(qleft[Hydro::ID0] + qleft[Hydro::ID1], smallr);
  auto       pl = fmax(qleft[Hydro::IP], rl * smallp);
  const auto ul = qleft[Hydro::IU];

  const auto eintl = eos.mixture_specific_eint(
    rl, pl, qleft[Hydro::IPHI], 1 - qleft[Hydro::IPHI], qleft[Hydro::ID0], qleft[Hydro::ID1]);
  pl = eos.mixture_pressure(
    rl, eintl, qleft[Hydro::IPHI], 1 - qleft[Hydro::IPHI], qleft[Hydro::ID0], qleft[Hydro::ID1]);

  auto ecinl = HALF_F * rl * ul * ul;
  ecinl += HALF_F * rl * qleft[Hydro::IV] * qleft[Hydro::IV];
  if constexpr (dim == 3)
    ecinl += HALF_F * rl * qleft[Hydro::IW] * qleft[Hydro::IW];

  const auto cl = eos.mixture_sound_speed(
    rl, pl, qleft[Hydro::IPHI], 1 - qleft[Hydro::IPHI], qleft[Hydro::ID0], qleft[Hydro::ID1]);
  auto etotl = rl * eintl + ecinl;
  auto ptotl = pl;

  //
  // Right variables
  //
  const auto rr = fmax(qright[Hydro::ID0] + qright[Hydro::ID1], smallr);
  auto       pr = fmax(qright[Hydro::IP], rr * smallp);
  const auto ur = qright[Hydro::IU];

  const auto eintr = eos.mixture_specific_eint(
    rr, pr, qright[Hydro::IPHI], 1 - qright[Hydro::IPHI], qright[Hydro::ID0], qright[Hydro::ID1]);
  pr = eos.mixture_pressure(rr,
                            eintr,
                            qright[Hydro::IPHI],
                            1 - qright[Hydro::IPHI],
                            qright[Hydro::ID0],
                            qright[Hydro::ID1]);

  auto ecinr = HALF_F * rr * ur * ur;
  ecinr += HALF_F * rr * qright[Hydro::IV] * qright[Hydro::IV];
  if constexpr (dim == 3)
    ecinr += HALF_F * rr * qright[Hydro::IW] * qright[Hydro::IW];

  const auto cr = eos.mixture_sound_speed(
    rr, pr, qright[Hydro::IPHI], 1 - qright[Hydro::IPHI], qright[Hydro::ID0], qright[Hydro::ID1]);
  auto etotr = rr * eintr + ecinr;
  auto ptotr = pr;

  // Find the largest eigenvalues in the normal direction to the interface
  // and compute wave speed
  auto cfastl = sqrt(fmax(cl * cl, smallc * smallc));
  auto cfastr = sqrt(fmax(cr * cr, smallc * smallc));
  auto SL = fmin(ul, ur) - fmax(cfastl, cfastr);
  auto SR = fmax(ul, ur) + fmax(cfastl, cfastr);

  // wave speed by Garrick et al, JCP 344 (2017) 260-280
  // Garrick et al, JCP 344 (2017) 260-280
  // const auto u_avg = (ul + ur) * HALF_F;
  // const auto c_avg = (cl + cr) * HALF_F;
  // auto SL = fmin(u_avg - c_avg, ul - cl);
  // auto SR = fmax(u_avg + c_avg, ur + cr);

  // wave speed by Zhang et al., JCP 498 (2024) 112672
  // const auto sqrt_rl = sqrt(rl);
  // const auto sqrt_rr = sqrt(rr);
  // const auto u_avg = (ul * sqrt_rl + ur * sqrt_rr) / (sqrt_rl + sqrt_rr);
  // const auto c2_avg = (cl * cl * sqrt_rl + cr * cr * sqrt_rr) / (sqrt_rl + sqrt_rr) +
  //                     HALF_F * (sqrt_rl * sqrt_rr) / (sqrt_rl + sqrt_rr) / (sqrt_rl + sqrt_rr) *
  //                       (ur - ul) * (ur - ul);
  // const auto c_avg = sqrt(c2_avg);
  // auto       SL = fmin(u_avg - c_avg, ul - cl);
  // auto       SR = fmax(u_avg + c_avg, ur + cr);

  // Rusanov wave speed
  // const auto tmp1 = fmax(fabs(ul-cl),fabs(ur-cr));
  // const auto tmp2 = fmax(fabs(ul+cl),fabs(ur+cr));
  // auto SL = -fmax(tmp1,tmp2);
  // auto SR =  fmax(tmp1,tmp2);

  // Compute lagrangian sound speed
  auto rcl = rl * (ul - SL);
  auto rcr = rr * (SR - ur);

  // Compute acoustic star state
  ustar = (rcr * ur + rcl * ul + (ptotl - ptotr)) / (rcr + rcl);
  auto ptotstar = (rcr * ptotl + rcl * ptotr + rcl * rcr * (ul - ur)) / (rcr + rcl);

  // Left star region variables
  const auto factorL = (SL - ul) / (SL - ustar);
  // const auto rstarl = rl * factorL;
  const auto etotstarl = ((SL - ul) * etotl - ptotl * ul + ptotstar * ustar) / (SL - ustar);

  // Right star region variables
  const auto factorR = (SR - ur) / (SR - ustar);
  // const auto rstarr = rr * factorR;
  const auto etotstarr = ((SR - ur) * etotr - ptotr * ur + ptotstar * ustar) / (SR - ustar);

  // Sample the solution at x/t=0
  real_t ro, ro_0, ro_1, uo, ptoto, etoto;
  // auto vo;
  if (SL > ZERO_F)
  {
    ro_0 = qleft[Hydro::ID0];
    ro_1 = qleft[Hydro::ID1];
    ro = rl;
    uo = ul;
    // vo = vl;
    ptoto = ptotl;
    etoto = etotl;
  }
  else if (ustar > ZERO_F)
  {
    ro_0 = qleft[Hydro::ID0] * factorL;
    ro_1 = qleft[Hydro::ID1] * factorL;
    ro = rl * factorL;
    uo = ustar;
    // vo = vl * factorL;
    ptoto = ptotstar;
    etoto = etotstarl;
  }
  else if (SR > ZERO_F)
  {
    ro_0 = qright[Hydro::ID0] * factorR;
    ro_1 = qright[Hydro::ID1] * factorR;
    ro = rr * factorR;
    uo = ustar;
    // vo = vr * factorR;
    ptoto = ptotstar;
    etoto = etotstarr;
  }
  else
  {
    ro_0 = qright[Hydro::ID0];
    ro_1 = qright[Hydro::ID1];
    ro = rr;
    uo = ur;
    // vo = vr;
    ptoto = ptotr;
    etoto = etotr;
  }

  // Compute the Godunov flux
  flux[Hydro::ID0] = ro_0 * uo;
  flux[Hydro::ID1] = ro_1 * uo;
  flux[Hydro::ID] = (ro_0 + ro_1) * uo; // ro * uo;
  flux[Hydro::IU] = ro * uo * uo + ptoto;
  // flux[Hydro::IV] = ro * uo * vo;
  flux[Hydro::IP] = (etoto + ptoto) * uo;

  if (flux[Hydro::ID] > ZERO_F)
  {
    flux[Hydro::IV] = flux[Hydro::ID] * qleft[Hydro::IV];
  }
  else
  {
    flux[Hydro::IV] = flux[Hydro::ID] * qright[Hydro::IV];
  }

  if constexpr (dim == 3)
  {
    if (flux[Hydro::ID] > ZERO_F)
    {
      flux[Hydro::IW] = flux[Hydro::ID] * qleft[Hydro::IW];
    }
    else
    {
      flux[Hydro::IW] = flux[Hydro::ID] * qright[Hydro::IW];
    }
  }

  if (flux[Hydro::ID] > ZERO_F)
  {
    phistar = qleft[Hydro::IPHI];
  }
  else
  {
    phistar = qright[Hydro::IPHI];
  }

  ustar = uo;

  return riemann_state;

} // riemann_hllc

// =====================================================================================
// =====================================================================================
/**
 * Riemann solver HLLC - compact form
 *
 * \param[in] qL       : input left  state (primitive variables)
 * \param[in] qR       : input right state (primitive variables)
 * \param[out] flux    : output flux
 * \param[out] ustar   : velocity
 * \param[out] phistar : volumic fraction
 *
 *
 * Reference:
 * - http://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf
 * - "On the Choice of Wavespeeds for the HLLC Riemann Solver", Batten et al., SIAM J. Sci. Comp.,
 * vol 18, issue 6, 1997, https://doi.org/10.1137/S1064827593260140
 * - HLLC for multiphase flow with surface tension:
 *   "A hybrid WENO5IS-THINC reconstruction scheme for compressible multiphase flows", Zhang et al,
 *   JCP, 498 (2024) 112672. https://doi.org/10.1016/j.jcp.2023.112672
 *
 */
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION RiemannState<dim>
                       riemann_hllc_compact(HydroState<dim> const &        qL,
                                            HydroState<dim> const &        qR,
                                            HydroSettings const &          settings,
                                            EosWrapper_t<device_t> const & eos)
{
  RiemannState<dim> riemann_state;
  auto &            flux = riemann_state.flux;
  auto &            ustar = riemann_state.ustar;
  auto &            phistar = riemann_state.phistar;

  // makes enum Hydro::VarId available
  using Hydro = kalypsso::godunov_five_eq::models::FiveEq;

  auto const &                  smallr = settings.smallr;
  auto const &                  smallp = settings.smallp;
  [[maybe_unused]] auto const & smallc = settings.smallc;

  //
  // Left variables
  //
  const auto rL = fmax(qL[Hydro::ID0] + qL[Hydro::ID1], smallr);
  auto       pL = fmax(qL[Hydro::IP], rL * smallp);
  const auto uL = qL[Hydro::IU];

  const auto eintL = eos.mixture_specific_eint(
    rL, pL, qL[Hydro::IPHI], 1 - qL[Hydro::IPHI], qL[Hydro::ID0], qL[Hydro::ID1]);
  pL = eos.mixture_pressure(
    rL, eintL, qL[Hydro::IPHI], 1 - qL[Hydro::IPHI], qL[Hydro::ID0], qL[Hydro::ID1]);

  auto ecinL = HALF_F * rL * uL * uL;
  ecinL += HALF_F * rL * qL[Hydro::IV] * qL[Hydro::IV];
  if constexpr (dim == 3)
    ecinL += HALF_F * rL * qL[Hydro::IW] * qL[Hydro::IW];

  const auto cL = eos.mixture_sound_speed(
    rL, pL, qL[Hydro::IPHI], 1 - qL[Hydro::IPHI], qL[Hydro::ID0], qL[Hydro::ID1]);
  const auto etotL = rL * eintL + ecinL;

  //
  // Right variables
  //
  const auto rR = fmax(qR[Hydro::ID0] + qR[Hydro::ID1], smallr);
  auto       pR = fmax(qR[Hydro::IP], rR * smallp);
  const auto uR = qR[Hydro::IU];

  const auto eintR = eos.mixture_specific_eint(
    rR, pR, qR[Hydro::IPHI], 1 - qR[Hydro::IPHI], qR[Hydro::ID0], qR[Hydro::ID1]);
  pR = eos.mixture_pressure(
    rR, eintR, qR[Hydro::IPHI], 1 - qR[Hydro::IPHI], qR[Hydro::ID0], qR[Hydro::ID1]);

  auto ecinR = HALF_F * rR * uR * uR;
  ecinR += HALF_F * rR * qR[Hydro::IV] * qR[Hydro::IV];
  if constexpr (dim == 3)
    ecinR += HALF_F * rR * qR[Hydro::IW] * qR[Hydro::IW];

  const auto cR = eos.mixture_sound_speed(
    rR, pR, qR[Hydro::IPHI], 1 - qR[Hydro::IPHI], qR[Hydro::ID0], qR[Hydro::ID1]);
  const auto etotR = rR * eintR + ecinR;

  // Find the largest eigenvalues in the normal direction to the interface
  const auto cfastL = sqrt(fmax(cL * cL, smallc * smallc));
  const auto cfastR = sqrt(fmax(cR * cR, smallc * smallc));

  //
  // Compute HLL wave speeds
  //

  // naive version
  const auto SL = fmin(uL, uR) - fmax(cfastL, cfastR);
  const auto SR = fmax(uL, uR) + fmax(cfastL, cfastR);

  // wave speed by Garrick et al, JCP 344 (2017) 260-280
  // Garrick et al, JCP 344 (2017) 260-280
  // const auto u_avg = (uL + uR) * HALF_F;
  // const auto c_avg = (cL + cR) * HALF_F;

  // const auto SL = fmin(u_avg - c_avg, uL - cL);
  // const auto SR = fmax(u_avg + c_avg, uR + cR);

  const auto u_star =
    (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR));
  // const auto p_star = pL + rL * (SL - uL) * (u_star - uL);

  const auto factorL = (SL - uL) / (SL - u_star);
  const auto factorR = (SR - uR) / (SR - u_star);

  const auto aL = (1 + COPYSIGN(ONE_F, u_star)) * HALF_F;
  const auto aR = (1 - COPYSIGN(ONE_F, u_star)) * HALF_F;

  const auto Sm = fmin(ZERO_F, SL);
  const auto Sp = fmax(ZERO_F, SR);

  HydroState<dim> UstarL, UstarR;

  UstarL[Hydro::ID0] = factorL * qL[Hydro::ID0];
  UstarL[Hydro::ID1] = factorL * qL[Hydro::ID1];
  UstarL[Hydro::ID] = factorL * rL;
  UstarL[Hydro::IU] = factorL * rL * u_star;
  UstarL[Hydro::IV] = factorL * rL * qL[Hydro::IV];
  if constexpr (dim == 3)
  {
    UstarL[Hydro::IW] = factorL * rL * qL[Hydro::IW];
  }
  UstarL[Hydro::IE] = factorL * (etotL + (u_star - uL) * (rL * u_star + (pL / (SL - uL))));

  UstarR[Hydro::ID0] = factorR * qR[Hydro::ID0];
  UstarR[Hydro::ID1] = factorR * qR[Hydro::ID1];
  UstarR[Hydro::ID] = factorR * rR;
  UstarR[Hydro::IU] = factorR * rR * u_star;
  UstarR[Hydro::IV] = factorR * rR * qR[Hydro::IV];
  if constexpr (dim == 3)
  {
    UstarR[Hydro::IW] = factorR * rR * qR[Hydro::IW];
  }
  UstarR[Hydro::IE] = factorR * (etotR + (u_star - uR) * (rR * u_star + (pR / (SR - uR))));

  HydroState<dim> fL, fR;
  fL[Hydro::ID0] = qL[Hydro::ID0] * uL;
  fL[Hydro::ID1] = qL[Hydro::ID1] * uL;
  fL[Hydro::ID] = fL[Hydro::ID0] + fL[Hydro::ID1];
  fL[Hydro::IU] = rL * uL * uL + pL;
  fL[Hydro::IV] = rL * qL[Hydro::IV] * uL;
  if constexpr (dim == 3)
  {
    fL[Hydro::IW] = rL * qL[Hydro::IW] * uL;
  }
  fL[Hydro::IE] = (etotL + pL) * uL;

  fR[Hydro::ID0] = qR[Hydro::ID0] * uR;
  fR[Hydro::ID1] = qR[Hydro::ID1] * uR;
  fR[Hydro::ID] = fR[Hydro::ID0] + fR[Hydro::ID1];
  fR[Hydro::IU] = rR * uR * uR + pR;
  fR[Hydro::IV] = rR * qR[Hydro::IV] * uR;
  if constexpr (dim == 3)
  {
    fR[Hydro::IW] = rR * qR[Hydro::IW] * uR;
  }
  fR[Hydro::IE] = (etotR + pR) * uR;

  // Compute the Godunov flux
  flux[Hydro::ID0] = aL * (fL[Hydro::ID0] + Sm * (UstarL[Hydro::ID0] - qL[Hydro::ID0])) +
                     aR * (fR[Hydro::ID0] + Sp * (UstarR[Hydro::ID0] - qR[Hydro::ID0]));
  flux[Hydro::ID1] = aL * (fL[Hydro::ID1] + Sm * (UstarL[Hydro::ID1] - qL[Hydro::ID1])) +
                     aR * (fR[Hydro::ID1] + Sp * (UstarR[Hydro::ID1] - qR[Hydro::ID1]));
  flux[Hydro::ID] = flux[Hydro::ID0] + flux[Hydro::ID1];
  flux[Hydro::IU] = aL * (fL[Hydro::IU] + Sm * (UstarL[Hydro::IU] - rL * uL)) +
                    aR * (fR[Hydro::IU] + Sp * (UstarR[Hydro::IU] - rR * uR));
  flux[Hydro::IV] = aL * (fL[Hydro::IV] + Sm * (UstarL[Hydro::IV] - rL * qL[Hydro::IV])) +
                    aR * (fR[Hydro::IV] + Sp * (UstarR[Hydro::IV] - rR * qR[Hydro::IV]));
  if constexpr (dim == 3)
  {
    flux[Hydro::IW] = aL * (fL[Hydro::IW] + Sm * (UstarL[Hydro::IW] - rL * qL[Hydro::IW])) +
                      aR * (fR[Hydro::IW] + Sp * (UstarR[Hydro::IW] - rR * qR[Hydro::IW]));
  }
  flux[Hydro::IE] = aL * (fL[Hydro::IE] + Sm * (UstarL[Hydro::IE] - etotL)) +
                    aR * (fR[Hydro::IE] + Sp * (UstarR[Hydro::IE] - etotR));

  if (flux[Hydro::ID] > ZERO_F)
  {
    phistar = qL[Hydro::IPHI];
  }
  else
  {
    phistar = qR[Hydro::IPHI];
  }

  if (SL > ZERO_F)
  {
    ustar = uL;
  }
  else if (u_star > ZERO_F)
  {
    ustar = u_star;
  }
  else if (SR > ZERO_F)
  {
    ustar = u_star;
  }
  else
  {
    ustar = uR;
  }

  return riemann_state;

} // riemann_hllc_compact

/**
 * Wrapper function calling the actual riemann solver.
 */
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION RiemannState<dim>
                       riemann_hydro(HydroState<dim> const &        qleft,
                                     HydroState<dim> const &        qright,
                                     HydroSettings const &          settings,
                                     EosWrapper_t<device_t> const & eos)
{

  if (settings.riemannSolverType == +RiemannSolverType::HLLC)
  {
    return riemann_hllc<dim>(qleft, qright, settings, eos);
    // return riemann_hllc_compact<dim>(qleft, qright, settings, eos);
  }
  else
  {
    Kokkos::abort("Unknown Riemann solver type. Please update your input parameter file, only HLLC "
                  "is supported here");
  }

} // riemann_hydro

} // namespace godunov_five_eq

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_FIVE_EQ_MODELS_RIEMANN_SOLVERS_H_
