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

    Note that wiggle() optionally returns the figure instance. It is adviced 
    to use the column major order for array as in Fortran to optimal 
    performance.

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


# def sta_lta(trace, stw, ltw):
#     ''' STA/LTA ratio for first break detection

#     Keyword arguments:
#     trace:      Input data trace
#     stw:        Short-time average window length
#     ltw:        Long-time average window length

#     Returns:
#     r:      STA/LTA ratio curve
#     d:      STA/LTA ratio derivative curve

#     Reference: 
#     http://www.crewes.org/ForOurSponsors/ConferenceAbstracts/2009/CSEG/Wong_CSEG_2009.pdf
#     '''

#     # Input check
#     if len(trace.shape) > 1:
#         raise ValueError('trace is a one-dimensional array')
#     if stw >= ltw:
#         raise ValueError("STW needs to be less than LTW")

#     Ns = len(trace)
#     trace = np.hstack((trace, np.mean(trace[:2]) * np.ones(ltw - 1)))
#     r = np.zeros(Ns)
#     for ns in range(Ns):
#         # computer sta/lta ratio
#         r[ns] = (np.mean(trace[range(ns - stw + 1, ns + 1)]**2)) /\
#             (np.mean(trace[range(ns - ltw + 1, ns + 1)]**2))
#     # compute the derivative of sta/lta ratio
#     d = np.hstack((np.diff(r), 0))
#     return r, d

if __name__ == '__main__':
    data = np.vstack((ricker()[1], ricker()[1])).T
    plt.figure()
    wiggle(data)
    plt.grid()
    # plt.figure()
    # plt.plot(sta_lta(ricker()[1], 5, 30)[0])
    # plt.figure()
    # plt.plot(sta_lta(ricker()[1], 5, 30)[1])
    plt.show()
    p = traces(data)
    p.setLabel('left', "Time", units='sec')
    p.setLabel('bottom', "Traces number", units='')
    p.showGrid(x=True, y=True)
    show()
