'''Detection module for Lijun's seismic toolbox'''
import numpy as np
import numba


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


@numba.jit(nopython=True)
def er(trace, Ne):
    '''Modified energy ratio for first break detection

    Keyword argument:
    trace:  Input data trace
    Ne:     energy window size

    Returns:
    er:     Energy ratio
    er3:    Modified energy ratio 3

    Parameter selection:
    Window Size(ne): 2–3 times the dominant period of the signal
    '''

    Nsp = len(trace)
    initial_value = (trace[0] + trace[1]) / 2
    end_value = (trace[-2] + trace[-1]) / 2
    # Energy ratio
    ratio = np.zeros(Nsp)
    for nsp in range(Nsp):
        E1 = 0
        for ne in range(nsp, nsp + Ne):
            if ne >= Nsp:
                E1 += end_value**2
            else:
                E1 += trace[ne]**2
        E2 = 0
        for ne in range(nsp - Ne + 1, nsp):
            if ne < 0:
                E2 += initial_value**2
            else:
                E2 += trace[ne]**2
        ratio[nsp] = E1 / E2

    return ratio


@numba.jit(nopython=True)
def eps(trace, Ne):
    ''' Edge-preserving smoothing

    Keyword argument:
    trace:  input data
    Ne:     window length

    Returns:
    trace_eps:   Edge preserved smoothed trace

    Parameter selection:
    Window length(Ne): times the dominant period of the signal

    Reference:
    Luo, Y.,M.Marhoon, S. A. Dossary, andM. Alfaraj, 2002, Edge-preserving smoothing and applications: The Leading Edge, 21, 136–158.
    '''

    Nsp = len(trace)
    initial_value = (trace[0] + trace[1]) / 2
    end_value = (trace[-2] + trace[-1]) / 2

    # Initialize output array
    trace_eps = np.zeros(Nsp)
    for nsp in range(Nsp):
        for ne in range(Ne):
            window = np.zeros(Ne)
            nc = 0
            for nw in range(nsp - Ne + 1 + ne, nsp + ne):
                if nw < 0:
                    window[nc] = initial_value
                elif nw >= Nsp:
                    window[nc] = end_value
                else:
                    window[nc] = trace[nw]
                nc += 1
            std = window.std()
            if ne == 0:
                std_min = std
                mean_target = window.mean()
            elif std < std_min:
                std_min = std
                mean_target = window.mean()
        trace_eps[nsp] = mean_target

    return trace_eps


@numba.jit(nopython=True)
def cm(trace, Nl, beta=0.2):
    ''' Coppens' method

    Keyword argument:
    trace:  input trace
    Nl:     leading window length
    beta:   demoninator factor

    Returns:
    ratio:  Coppens' method ratio

    Parameter selection:
    Leading window length(Nl): 1 time the dominant period of the signal

    Reference:
    Coppens, F. "FIRST ARRIVAL PICKING ON COMMON‐OFFSET TRACE COLLECTIONS FOR AUTOMATIC ESTIMATION OF STATIC CORRECTIONS*." Geophysical Prospecting 33.8 (1985): 1212-1231.

    '''

    Nsp = len(trace)

    # Coppen's method
    ratio = np.zeros(Nsp)
    for nsp in range(Nsp):
        E1 = 0
        for nl in range(nsp - Nl + 1, nsp + 1):
            if nl >= 0:
                E1 += trace[nl]**2
        E2 = 0
        for nl in range(nsp + 1):
            E2 += trace[nl]**2
        ratio[nsp] = E1 / (E2 + beta)
        # mcm[nsp] = np.sum(trace_ext[range(nsp - nl + 1, nsp + 1)]**2) /\
        #     (np.sum(trace_ext[range(nsp + 1)]**2) + beta)

    return ratio


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
