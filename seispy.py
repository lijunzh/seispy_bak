'''Lijun's personal toolbox for seismic processing'''
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


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


def insert_zeros(trace, tt=None):
    ''' Insert zero locations in data trace and tt vector based on linear fit

    '''

    if tt is None:
        tt = np.arange(len(trace))

    # Find zeros
    zc_idx = np.where(np.diff(np.signbit(trace)))[0]
    x1 = tt[zc_idx]
    x2 = tt[zc_idx + 1]
    y1 = trace[zc_idx]
    y2 = trace[zc_idx + 1]
    a = (y2 - y1) / (x2 - x1)
    tt_zero = x1 - y1 / a

    # split tt and trace
    tt_split = np.split(tt, zc_idx + 1)
    trace_split = np.split(trace, zc_idx + 1)
    tt_zi = tt_split[0]
    trace_zi = trace_split[0]

    # insert zeros in tt and trace
    for i in range(len(tt_zero)):
        tt_zi = np.hstack(
            (tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack(
            (trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi


def wiggle_input_check(data, tt, xx, sf, verbose):
    ''' Helper function for wiggle() and traces() to check input

    '''

    # Input check for verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    # Input check for data
    if type(data).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")

    if len(data.shape) != 2:
        raise ValueError("data must be a 2D array")

    # Input check for tt
    if tt is None:
        tt = np.arange(data.shape[0])
        if verbose:
            print("tt is automatically generated.")
            print(tt)
    else:
        if type(tt).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(tt.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")

    # Input check for xx
    if xx is None:
        xx = np.arange(data.shape[1])
        if verbose:
            print("xx is automatically generated.")
            print(xx)
    else:
        if type(xx).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(xx.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")

    # Input check for streth factor (sf)
    if not isinstance(sf, (int, float)):
        raise TypeError("Strech factor(sf) must be a number")

    # Compute trace horizontal spacing
    ts = np.min(np.diff(xx))

    # Rescale data by trace_spacing and strech_factor
    data_max_std = np.max(np.std(data, axis=0))
    data = data / data_max_std / ts * sf

    return data, tt, xx, ts


def wiggle(data, tt=None, xx=None, color='k', sf=0.15, verbose=False):
    '''Wiggle plot of a sesimic data section

    Syntax examples:
        wiggle(data)
        wiggle(data, tt)
        wiggle(data, tt, xx)
        wiggle(data, tt, xx, color)
        fi = wiggle(data, tt, xx, color, sf, verbose)

    Use the column major order for array as in Fortran to optimal performance.

    The following color abbreviations are supported:

    ==========  ========
    character   color
    ==========  ========
    'b'         blue
    'g'         green
    'r'         red
    'c'         cyan
    'm'         magenta
    'y'         yellow
    'k'         black
    'w'         white
    ==========  ========


    '''

    # Input check
    data, tt, xx, ts = wiggle_input_check(data, tt, xx, sf, verbose)

    # Plot data using matplotlib.pyplot
    Ntr = data.shape[1]

    ax = plt.gca()
    for ntr in range(Ntr):
        trace = data[:, ntr]
        offset = xx[ntr]

        trace_zi, tt_zi = insert_zeros(trace, tt)
        ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                         where=trace_zi >= 0,
                         facecolor=color)
        ax.plot(trace_zi + offset, tt_zi, color)

    ax.invert_yaxis()
    ax.set_xlim(xx[0] - ts, xx[-1] + ts)


def traces(data, tt=None, xx=None, color='k', sf=0.15, verbose=False):
    '''Plot large seismic dataset in real time using pyqtgraph

    '''

    # Input check
    data, tt, xx, ts = wiggle_input_check(data, tt, xx, sf, verbose)

    Ntr = data.shape[1]

    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    pg.setConfigOptions(antialias=True)  # Enable antialiasing

    p = pg.plot()

    for ntr in range(Ntr):
        trace = data[:, ntr]
        offset = xx[ntr]

        # Insert zeros
        trace_zi, tt_zi = insert_zeros(trace, tt)
        # Seperate top and bottom line
        trace_top = np.array(
            [i + offset if i >= 0 else None for i in trace_zi],
            dtype='float64')
        trace_line = np.array(
            [offset if i >= 0 else None for i in trace_zi],
            dtype='float64')
        trace_bot = np.array(
            [i + offset if i <= 0 else None for i in trace_zi],
            dtype='float64')
        # Plot top and bottom
        top = p.plot(x=trace_top, y=tt_zi, pen=color)
        bot = p.plot(x=trace_line, y=tt_zi, pen=color)
        p.plot(x=trace_bot, y=tt_zi, pen=color)
        fill = pg.FillBetweenItem(bot, top, brush=color)
        p.addItem(fill)

    p.invertY(True)
    p.setRange(yRange=[0, np.max(tt)], padding=0)

    return p


def show():
    '''Helper function to show pyqtgraph figres

    '''
    QtGui.QApplication.instance().exec_()


def slr(trace, ns, nl):
    '''STA/LTA ratio for first break detection

    Keyword arguments:
    trace:      Input data trace
    ns:        Short-time average window length
    nl:        Long-time average window length

    Returns:
    r:      STA/LTA ratio curve
    d:      STA/LTA ratio derivative curve

    Parameter selection:
    STA(ns): 2–3 times the dominant period of the signal
    LTA(nl): 5–10 times STA
    Reference: http://www.crewes.org/ForOurSponsors/ConferenceAbstracts/2009/CSEG/Wong_CSEG_2009.pdf
    '''

    # Input check
    if type(trace).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")
    if len(trace.shape) != 1:
        raise ValueError('trace is a one-dimensional array')
    if ns >= nl:
        raise ValueError("ns needs to be less than nl")

    Nsp = len(trace)    # number of sample points in trace
    # Extend trace for ns and nl
    trace_ext = np.hstack((trace,
                           np.mean(trace[-3:-1]) * np.ones(nl - 1),
                           np.mean(trace[:2]) * np.ones(nl - 1)))
    r = np.zeros(Nsp)
    for nsp in range(Nsp):
        # computer sta/lta ratio (index starting from 0)
        r[nsp] = (np.mean(trace_ext[range(nsp - ns + 1, nsp + 1)]**2)) /\
            (np.mean(trace_ext[range(nsp - nl + 1, nsp + 1)]**2))
    # compute the derivative of sta/lta ratio
    d = np.hstack((np.diff(r), 0))
    return r, d


def mer(trace, ne):
    '''Modified energy ratio for first break detection

    Keyword argument:
    trace:  Input data trace
    ne:     energy window size

    Returns:
    er:     Energy ratio
    er3:    Modified energy ratio 3

    Parameter selection:
    Window Size(ne): 2–3 times the dominant period of the signal
    '''

    # Input check
    if type(trace).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")
    if len(trace.shape) != 1:
        raise ValueError('trace is a one-dimensional array')
    if not isinstance(ne, (int, float)):
        raise TypeError("ne must be a number")
    if ne <= 0:
        raise ValueError("ne must be a possitive number")
    if ne >= len(trace):
        raise ValueError("ne must be less than length of trace")

    Nsp = len(trace)    # number of sample points in trace
    # Extend trace(both head and tail)
    trace_ext = np.hstack((trace,
                           np.mean(trace[-3:-1]) * np.ones(ne - 1),
                           np.mean(trace[:2]) * np.ones(ne - 1)))

    # Energy ratio
    er = np.zeros(Nsp)
    for nsp in range(Nsp):
        er[nsp] = np.sum(trace_ext[range(nsp, nsp + ne)]**2) /\
            np.sum(trace_ext[range(nsp - ne + 1, nsp + 1)]**2)

    er3 = np.power((np.abs(trace) * er), 3)
    return er, er3


def rolling_window(a, window, step=1):
    '''Rolling window of a

    Keyword argument:
    a:  Input 1D array
    window: Window length
    step:  steps in sample point

    Return:
    rolling windowed data

    '''
    shape = a.shape[:-1] + (int((a.shape[-1] - window) / step) + 1, window)
    strides = (a.strides[-1] * step,) + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def eps(trace, ne):
    ''' Edge-preserving smoothing

    Keyword argument:
    trace:  Input data
    ne:     Window length

    Returns:
    trace_eps:   Edge preserved smoothed trace

    Reference:
    Luo, Y.,M.Marhoon, S. A. Dossary, andM. Alfaraj, 2002, Edge-preserving smoothing and applications: The Leading Edge, 21, 136–158.
    '''

    Nsp = len(trace)

    # Extend trace
    trace_ext = np.hstack((trace,
                           np.mean(trace[-3:-1]) * np.ones(ne - 1),
                           np.mean(trace[:2]) * np.ones(ne - 1)))

    # Initialize output array
    trace_eps = np.zeros(trace.shape)
    for nsp in range(Nsp):
        windows = rolling_window(trace_ext[range(nsp - ne + 1, nsp + ne)], ne)
        selected_window_idx = np.argmin(np.std(windows, axis=1))
        trace_eps[nsp] = np.mean(windows[selected_window_idx])

    return trace_eps


def mcm(trace, nl, ne, beta=0.2, mcm=True):
    ''' Coppen's method and modified Coppen's method

    '''

    Nsp = len(trace)

    # EPS
    if mcm:
        trace_eps = eps(trace, ne)
    else:
        trace_eps = trace

    # Extend trace
    trace_ext = np.hstack((trace_eps, np.zeros(nl - 1)))

    # Coppen's method
    mcm = np.zeros(trace.shape)
    for nsp in range(Nsp):
        mcm[nsp] = np.sum(trace_ext[range(nsp - nl + 1, nsp + 1)]**2) /\
            (np.sum(trace_ext[range(nsp + 1)]**2) + beta)

    return mcm


def em():
    '''Entropy method

    '''
    pass


def fdm():
    '''Fractal-dimension method

    '''
    pass


def pai_k():
    '''PAI-K method

    '''
    pass


def slr_kurt():
    '''Short-term kurtosis over long-term kurtosis ratio

    '''
    pass


def aic():
    '''Akaike information criterion method

    '''
    pass


if __name__ == '__main__':
    # Rolling window
    print(rolling_window(np.arange(10), 3, 1))

    # EPS
    print(eps(np.arange(10), 3))

    # Test time picking methods
    dt = 0.002
    trace = ricker(f=10, len=10, dt=dt, peak_loc=5)[1]
    trace = trace + np.random.normal(loc=0., scale=0.2, size=trace.shape)
    plt.figure()
    plt.subplot(711)
    plt.plot(trace)
    plt.grid()
    plt.ylabel("trace")

    ns = int(1 * 0.1 / dt)
    nl = int(10 * 0.1 / dt)
    plt.subplot(712)
    plt.plot(slr(trace, ns, nl)[0])
    plt.grid()
    plt.ylabel("slr")
    plt.subplot(713)
    plt.plot(slr(trace, ns, nl)[1])
    plt.grid()
    plt.ylabel("dslr")

    ne = int(2 * 0.1 / dt)
    plt.subplot(714)
    plt.plot(mer(trace, ne)[0])
    plt.grid()
    plt.ylabel("er")
    plt.subplot(715)
    plt.semilogy(mer(trace, ne)[1])
    plt.grid()
    plt.ylabel("mer")

    nl = int(0.1 / dt)
    ne = int(1.5 * 0.1 / dt)
    plt.subplot(716)
    plt.plot(mcm(trace, nl, ne, mcm=False))
    plt.grid()
    plt.ylabel("cm")
    plt.subplot(717)
    plt.plot(mcm(trace, nl, ne, mcm=True))
    plt.grid()
    plt.ylabel("mcm")
    plt.show()

    # Test ricker
    # data = np.vstack((ricker()[1], ricker()[1])).T

    # Test wiggle()
    # plt.figure()
    # wiggle(data)
    # plt.grid()

    # Test traces()
    # p = traces(data)
    # p.setLabel('left', "Time", units='sec')
    # p.setLabel('bottom', "Traces number", units='')
    # p.showGrid(x=True, y=True)
    # show()
