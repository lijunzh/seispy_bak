# -*- coding: utf-8 -*-
"""Core module for Seispy toolbox

This is the core module of Seispy toolbox. Inside this module, only commonly
used objects are included.

Attributes:
    ricker(): function that generate a 1-D numpy array that represent a
    Ricker wavelet signal


Todo:
    * Re-organize package and move useful objects to core.py
    *
"""
import numpy as np


def ricker(f=10, len=0.5, dt=0.002, peak_loc=0.25):
    """Generate ricker wavelet signal for seismic simulation:

    Args:
    f (float): peak frequency in Hz (default 10)
    len(float): length of signal in sec (default 0.5)
    dt (float): time resolution in sec (default 0.002)
    peak_loc (float): peak location in sec (default 0.25)


    Returns:
    t (nparray): time sequence
    y (nparray): ricker signal sequence

    Examples:
        This function can be called directly with no argument as following::

            $ ricker_signal = seispy.ricker()

        which returns a 10 Hz Ricker wavelet that peaked at 0.25 sec,  0.5
        sec long and sampled at 500 Hz.

        Users can generate different Ricker wavelet by tweaking the input
        argument

    References:
        http://google.github.io/styleguide/pyguide.html

    """

    # Generate time sequence based on sample frequency/period, signal length
    # and peak location
    t = np.linspace(-peak_loc, len - peak_loc - dt, int(len / dt))

    # Shift signal to the correct location
    t_out = t + peak_loc  # time shift ricker wavelet based on peak_loc

    # Generate Ricker wavelet signal based on reference
    y = (1 - 2 * np.pi ** 2 * f ** 2 * t ** 2) * np.exp(
        -np.pi ** 2 * f ** 2 * t ** 2)

    return t_out, y


if __name__ == '__main__':
    pass
