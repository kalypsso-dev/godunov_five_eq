#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys, getopt
import argparse

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import configparser
import StiffenedGas as sg

def do_plot(prefix1, prefix2, shock_tube_name):

    t1   = np.load(prefix1+"_mixing_times.npy")
    t2   = np.load(prefix2+"_mixing_times.npy")
    mix1 = np.load(prefix1+"_mixing_integrals.npy")
    mix2 = np.load(prefix2+"_mixing_integrals.npy")

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(11,8))

    ax1.plot(t1, mix1, 'r-', label='no thinc')
    ax1.plot(t2, mix2, 'r--', label='thinc')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax1.set_xlabel('time', fontsize=16)
    ax1.set_ylabel('\large $\int_\Omega \phi(1-\phi)d\mathbf{r}$', fontsize=16)

    ax1.legend()
    plt.suptitle(r'Gas-Water shock tube : evolution of $\int_\Omega \phi(1-\phi)d\mathbf{r}$', fontsize=20)
    plt.show()

###############################################################################
if __name__ == "__main__":

    shock_tube_name="Gas-Water"
    parser = argparse.ArgumentParser(description='Display '+shock_tube_name+' plots.')
    parser.add_argument('--ini1', type=str, default='test_two_fluid_shock_tube_2d', help='prefix to ini parameter file 1')
    parser.add_argument('--ini2', type=str, default='test_two_fluid_shock_tube_2d_thinc', help='prefix to ini parameter file 2')
    args = parser.parse_args()

    do_plot(args.ini1, args.ini2, shock_tube_name)
