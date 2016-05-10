'''Detection module for Lijun's seismic toolbox'''
import numpy as np
import numba

__all__ = ['slr', 'er', 'eps', 'cm', 'em', 'fdm', 'pai_k', 'slr_kurt', 'aic']


@numba.jit(nopython=True)
def slr(trace, Ns, Nl):
    '''STA/LTA ratio for first break detection

    Keyword arguments:
    trace:      Input data trace
    Ns:         Short-time average window length
    Nl:         Long-time average window length
    derivative: output slr or derivative of slr

    Returns:
    r:      STA/LTA ratio curve
    d:      STA/LTA ratio derivative curve

    Parameter selection:
    STA(ns): 2–3 times the dominant period of the signal
    LTA(nl): 5–10 times STA
    Reference: http://www.crewes.org/ForOurSponsors/ConferenceAbstracts/2009/CSEG/Wong_CSEG_2009.pdf
    '''

    # Input check
    # if type(trace).__module__ != np.__name__:
    #     raise TypeError("data must be a numpy array")
    # if len(trace.shape) != 1:
    #     raise ValueError('trace is a one-dimensional array')
    # if Ns >= Nl:
    #     raise ValueError("ns needs to be less than nl")

    Nsp = len(trace)

    initial_value = (trace[0] + trace[1]) / 2
    ratio = np.zeros(Nsp)
    for nsp in range(Nsp):
        STA = 0
        for ns in range(nsp - Ns + 1, nsp + 1):
            if ns < 0:
                STA += initial_value**2
            else:
                STA += trace[ns]**2
        STA /= Ns
        LTA = 0
        for nl in range(nsp - Nl + 1, nsp + 1):
            if nl < 0:
                LTA += initial_value**2
            else:
                LTA += trace[nl]**2
        LTA /= Nl
        ratio[nsp] = STA / LTA
    return ratio


def er(trace, ne, mer=False):
    '''Modified energy ratio for first break detection

    Keyword argument:
    trace:  Input data trace
    ne:     energy window size
    mer:    output er or mer

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

    if mer:
        er3 = np.power((np.abs(trace) * er), 3)
        return er3
    else:
        return er


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


def cm(trace, nl, ne, beta=0.2, mcm=True):
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
