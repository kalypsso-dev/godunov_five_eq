#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys, getopt
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import configparser

from numpy import polyfit

def do_plot():

    # origin of time (first time incident shock hits the bubble)
    t0 = 0.058

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11,13))

    #
    # refracted wave
    #
    tr=np.array([0.070, 0.080, 0.090, 0.100])
    tr=tr-t0
    xr=np.array([205.6, 214.9, 224.7, 234.1])
    p, V = np.polyfit(tr, xr, 1, cov=True)
    print("v_refracted: {} +/- {}".format(p[0], np.sqrt(V[0][0])))
    xr_fit = tr*p[0]+p[1]

    #
    # jet
    #
    tj=np.array([0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180, 0.190, 0.200, 0.210, 0.220, 0.230, 0.240, 0.250, 0.260, 0.270, 0.280, 0.290, 0.300])
    tj=tj-t0
    xj=np.array([196.6, 198.3, 200.1, 202.0, 203.9, 205.9, 207.9, 209.8, 211.8, 213.8, 215.7, 217.7, 219.75, 221.8, 224.0, 226.0, 228.1, 230.4, 232.7, 235.1, 237.5, 239.8, 242.2, 244.65 ])

    #p, V = np.polyfit(tj, xj, 1, cov=True)
    #print("v_jet: {} +/- {}".format(p[0], np.sqrt(V[0][0])))
    # jet velocity should be measured after t=0.170
    p, V = np.polyfit(tj[10:], xj[10:], 1, cov=True)
    print("v_jet: {} +/- {}".format(p[0], np.sqrt(V[0][0])))

    #
    # downstream interface (uncertainty about +/- 0.1)
    #
    tdi=np.array([0.060, 0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180, 0.190, 0.200, 0.210, 0.220, 0.230, 0.240, 0.250, 0.260, 0.270, 0.280, 0.290, 0.300])
    tdi=tdi-t0
    xdi=np.array([244.96, 244.96, 244.96, 244.96, 244.96, 244.96, 245.5, 246.3, 247.25, 248.25, 249.25, 250.33, 251.47, 252.7, 253.96, 255.3, 256.6, 258.2, 259.7, 261.25, 262.7, 264.22, 265.7, 267.27, 268.7])
    p, V = np.polyfit(tdi[10:], xdi[10:], 1, cov=True)
    print("v_di: {} +/- {}".format(p[0], np.sqrt(V[0][0])))

    #
    # upstream interface (not very well defined - not measured)
    #
    #tui=np.array([])
    #xui=np.array([])

    #
    # shock
    #
    ts=np.array([0.060, 0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140])
    ts=ts-t0
    xs=np.array([195.2, 199.4, 203.72, 207.97, 212.2, 216.38, 220.63, 224.79, 229.08])
    p, V = np.polyfit(ts, xs, 1, cov=True)
    print("v_shock: {} +/- {}".format(p[0], np.sqrt(V[0][0])))

    # transmitted
    tt=np.array([0.120, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180, 0.190, 0.200, 0.210, 0.220, 0.230, 0.240, 0.250, 0.260, 0.270, 0.280, 0.290, 0.300])
    tt=tt-t0
    xt=np.array([248.2, 252.0, 255.8, 259.65, 263.4, 267.2, 271.02, 274.8, 278.6, 282.3, 286.2, 290.05, 293.9, 297.05, 301.55, 305.42, 309.25, 313.1, 316.93])
    p, V = np.polyfit(tt, xt, 1, cov=True)
    print("v_transmitted: {} +/- {}".format(p[0], np.sqrt(V[0][0])))

    # ax.plot(tr, xr, marker='v', label='refracted')
    # ax.plot(tj, xj, marker='s', label='jet')
    # ax.plot(tdi, xdi, marker='o', label='downstream interface')
    # ax.plot(tt, xt, marker='^', label='transmitted')
    # ax.plot(ts, xs, marker='+', label='shock')

    ax.plot(xr, tr, marker='v', linestyle='', linewidth=22, color='tab:blue', label='refracted')
    #ax.plot(xr_fit, tr, marker='v', linestyle='', linewidth=22, color='tab:blue', label='refracted - fit ')
    ax.plot(xj, tj, marker='s', linestyle='', linewidth=22, color='tab:orange', label='jet')
    ax.plot(xdi, tdi, marker='o', linestyle='', linewidth=22, color='tab:green', label='downstream interface')
    ax.plot(xt, tt, marker='^', linestyle='', linewidth=22, color='tab:red', label='transmitted')
    ax.plot(xs, ts, marker='x', markersize=10, linestyle='', linewidth=22, color='tab:purple', label='shock')

    ax.minorticks_on()

    tsize=22
    ax.tick_params(labelsize=tsize)
    ax.set_xlabel("x (mm)",size=28)
    ax.set_ylabel("t (ms)",size=28)

    lsize=24
    ax.legend(prop={'size':lsize})
    plt.suptitle('Time evolution of characteristic wave fronts', fontsize=32)
    plt.show()

###############################################################################
if __name__ == "__main__":

    do_plot()
