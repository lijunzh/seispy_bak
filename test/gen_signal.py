'''gen_signal.py generates common singals for testing

This program is part of Lijun Zhu's personal testing toolbox. 
It depends on the latest version following packages:
numpy
scipy
matplotlib

This module includes:
ricker(): generate ricker wavelet for seismic testing

For further information, contact: gatechzhu@gmail.com
'''

import numpy as np

__author__ = 'Lijun Zhu (gatechzhu@gmail.com)'
__version__ = '0.0.1'
__date__ = 'Mon Apr 11 16:42:46 2016'
__copyright__ = 'Copyright (c) 2016 Lijun Zhu'
__license__ = 'BSD-new license'


def ricker(f=10, len=0.5, dt=0.002, peak_loc=0.25):
    '''Generate ricker wavelet signal for seismic simulation:

    Keyword arguments:
    f -- (float) peak frequency in Hz (default 10)
    len -- (float) length of signal in sec (default 0.5)
    dt -- (float) time resolution in sec (default 0.002)
    peak_loc -- (float) peak location in sec (default 0.25)

    Returns:
    t -- (array) time vecotr
    y -- (array) signal vector
    '''

    t = np.linspace(-peak_loc, len - peak_loc - dt, int(len / dt))
    t_out = t + peak_loc
    y = (1 - 2 * np.pi**2 * f**2 * t**2) * np.exp(-np.pi**2 * f**2 * t**2)
    return t_out, y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # ricker()
    t, y = ricker()
    plt.plot(t, y)
    plt.show()
