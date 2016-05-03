'''Lijun's personal toolbox for seismic processing'''
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg


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

    if tt:
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


def wiggle(data, tt=None, xx=None, plot_type='matplotlib', options=None):
    '''Wiggle plot of a seismic data matrix

    Keyword arguments:
    data -- (2D ndarray) seismic data matrix
    tt -- (1D ndarray) time vector
    plot_type -- (string) type of plot (matplotlib or pyqtgraph)
    options -- (dict) options for wiggle

    Returns:
    fig/p -- figure handle depend on options

    options is a dict that is default for rmatplotlib as
    options = {
        'verbose': True,
        'shade': True,
        'orientation': 'vertical',
        'color': 'k',
        'symbols': '-',
        'trace_spacing': None,
        'rescale_factor': 0.2,
    }
    and for matplotlib as
    options = {
        'verbose': True,
        'shade': False,
        'orientation': 'vertical',
        'color': 'k',
        'trace_spacing': None,
        'rescale_factor': 0.2,
    }

    '''

    # Helper nested functions

    def input_check():
        '''Code block to check input data

        '''

        nonlocal data, tt, xx, plot_type, options

        # check 2D array
        if len(data.shape) is 2:
            Ns, Ntr = data.shape
        else:
            raise ValueError("data requires a 2D array!")

        # If C array, convert to F array
        # if data.flags['C_CONTIGUOUS']:
        #    print("Convert data array to Fortran order ...")
        #    data = np.asfortranarray(data)
        #    print("Done.")

        # check if tt is supplied as 1D array, generate it if not supplied
        if tt is None:
            tt = np.linspace(start=0, stop=Ns, num=Ns, endpoint=False)
        elif len(tt.shape) is not 1:
            raise ValueError("tt requires a 1D array!")

        # check if xx is supplied as 1D array, generate if if not supplied
        if xx is None:
            xx = np.linspace(start=0, stop=Ntr, num=Ntr, endpoint=False)
        elif len(xx.shape) is not 1:
            raise ValueError("xx requires a 1D array")

        # check if type is known
        available_plot_type_list = {'matplotlib', 'pyqtgraph'}
        if plot_type not in available_plot_type_list:
            raise ValueError("Unknown plot type!")

    # initialize options dict
    # Comment: it is possible to save all options in a json file so that
    # the default behavior of wiggle can be changed outside the source
    # code. However, it will sacrifice the startup time but the change of
    # these default values seldom happens.
    def options_init_matplotlib():
        '''Code block that initiate options dict for matplotlib plot type

        '''
        nonlocal options

        options = {
            'verbose': True,
            'shade': True,
            'orientation': 'vertical',
            'color': 'k',
            'symbols': '-',
            'trace_spacing': None,
            'rescale_factor': 0.1
        }

    def options_init_pyqtgraph():
        '''Code block that initiate options dict for pyqtgraph plot type

        '''
        nonlocal options

        options = {
            'verbose': True,
            'shade': False,
            'orientation': 'vertical',
            'color': 'k',
            'trace_spacing': None,
            'rescale_factor': 0.1
        }

    # Initialize options dict based on specified plot type
    def options_init(plot_type):
        '''Wrapper function for switch block of options_init based on plot_type

        '''

        return {
            'matplotlib': options_init_matplotlib,
            'pyqtgraph': options_init_pyqtgraph,
        }.get(plot_type, options_init_matplotlib)    # defualt matplot

    def calc_trace_spacing(xx):
        '''Calcuate trace spacing based on the input xx

        '''
        nonlocal options

        if not options['trace_spacing']:
            options['trace_spacing'] = np.min(np.diff(xx))

    def notify_user(plot_type, options):
        '''Notice user the detailed options used in wiggle plot

        '''

        print("Ploting data matrix using {:s}".format(plot_type))
        print("....detailed options:")
        for key, value in options.items():
            print("\t\t{:15s}| \t{}".format(key, value))

    # Performance concern: it is faster to apply array-wise operation on whole
    # array than access each column by for-loop.
    def rescale_data(options):
        '''Rescale the data matrix by trace_spacing and resacle_factor

        '''

        nonlocal data
        data = data / np.max(np.std(data, axis=0)) / \
            options['trace_spacing'] * options['rescale_factor']

    # data_with_zeros array generator for shade plots
    def insert_zeros(trace, tt):
        ''' Generate data and tt with zeros inserted for shade plots

        '''

        Ntr = data.shape[1]
        ntr = 0
        while ntr < Ntr:
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

    # We grouoped the functions that operate on each column of the array. It
    # is possible to de-couple the non-shade plot from shade plot; however, it
    # requires to use for-loop to loop through columns twice. One solution
    # used here is to generator
    def mplot():
        '''Plot data matrix using matplotlib with options

        '''

        nonlocal options

        Ntr = data.shape[1]

        ax = plt.gca()
        for ntr in range(Ntr):
            trace = data[:, ntr]
            offset = xx[ntr]

            if options['shade']:
                trace_zi, tt_zi = insert_zeros(trace, tt)
                ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                                 where=trace_zi >= 0,
                                 facecolor=options['color'])
                ax.plot(trace_zi + offset, tt_zi,
                        options['color'] + options['symbols'])
            else:
                ax.plot(trace + offset, tt,
                        options['color'] + options['symbols'])

        ax.invert_yaxis()
        return None

    def qtplot():
        '''Plot data matrix using pyqtgraph with options

        '''

        nonlocal data, tt, xx, options

        Ntr = data.shape[1]

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)  # Enable antialiasing

        pw = pg.plot()

        for ntr in range(Ntr):
            trace = data[:, ntr]
            offset = xx[ntr]

            if options['shade']:
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
                top = pw.plot(x=trace_top, y=tt_zi, pen=options['color'])
                line = pw.plot(x=trace_line, y=tt_zi, pen=options['color'])
                bot = pw.plot(x=trace_bot, y=tt_zi, pen=options['color'])
                fill = pg.FillBetweenItem(line, top, brush=options['color'])
                pw.addItem(fill)
            else:
                pw.plot(trace + offset, tt,
                        pen=options['color'])

        pw.setRange(yRange=[0, np.max(tt)], padding=0)
        pw.invertY(True)

        return pw

    def plot_data(plot_type):
        '''Wrapper function chooseing plotfun by plot_type

        '''

        return {
            'matplotlib': mplot,
            'pyqtgraph': qtplot,
        }.get(plot_type, mplot)

    # Main body of wiggle plot function
    input_check()

    if options is None:
        options_init(plot_type)()

    calc_trace_spacing(xx)

    if options['verbose']:
        notify_user(plot_type, options)

    rescale_data(options)

    return plot_data(plot_type)()


# def sta_lta(trace, stw, ltw):
#     ''' STA/LTA ratio for first break detection

#     Keyword arguments:
#     trace:      Input data trace
#     stw:        Short-time average window length
#     ltw:        Long-time average window length

#     Returns;
#     ratio:      STA/LTA ratio curve
#     '''

#     # Input check
#     if len(trace.shape) > 1:
#         raise ValueError('trace is a one-dimensional array')
#     if stw >= ltw:
#         raise ValueError("STW needs to be less than LTW")

#     ratio = np.zeros(trace.shape)
#     for nsample in range(1, len(trace) + 1):   # for every sample point
#         if nsample < ltw:
#             ratio[nsample - 1] = 0
#         else:
#             ratio[nsample - 1] = np.mean(trace[nsample - stw:nsample]) /\
#                 np.mean(trace[nsample - ltw:nsample])

#     return ratio

# if __name__ == '__main__':
#     print(sta_lta(np.arange(10), 2, 4))
