'''Helper module for common signal processing utilities'''
import numpy as np

# List of content
__all__ = ['ricker']


def ricker(f=10, len=0.5, dt=0.002, peak_loc=0.25):
    '''Generate ricker wavelet signal for seismic simulation:

    Keyword arguments:
    f -- (float) peak frequency in Hz (default 10)
    len -- (float) length of signal in sec (default 0.5)
    dt -- (float) time resolution in sec (default 0.002)
    peak_loc -- (float) peak location in sec (default 0.25)

    ------
    Returns:
    t -- (array) time vecotr
    y -- (array) signal vector

    '''

    t = np.linspace(-peak_loc, len - peak_loc - dt, int(len / dt))
    t_out = t + peak_loc    # time shift ricker wavelet based on peak_loc
    y = (1 - 2 * np.pi**2 * f**2 * t**2) * np.exp(-np.pi**2 * f**2 * t**2)
    return t_out, y
