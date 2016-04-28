# -*- coding: utf-8 -*-
''' Testing Python cabability of doing seismic data processing

'''

# add pyqtgraph path, need it for pyqt5 support
import sys
import numpy as np
import scipy.io as sio
import scipy.interpolate as sinterp
from scipy import signal
import matplotlib.pyplot as plt
import time
sys.path.insert(0, '/Users/lijun/Dropbox/tools/pytools/pyqtgraph')
sys.path.insert(0, '/Users/lijun/Dropbox/tools/pytools/seispy')
#sys.path.insert(0, '/home/lijun/Dropbox/tools/pytools/pyqtgraph')
#sys.path.insert(0, '/home/lijun/Dropbox/tools/pytools/seispy')

#import PyQt5
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import seispy as sp

mat = sio.loadmat('data/RealSeisData.mat')
real_data = mat['RealSeisData'][0, 0]['traces']
fs = mat['RealSeisData'][0, 0]['fs'][0, 0]
dt = 1 / fs

# Use 4th traces as template
real_template = real_data[:, 3]
real_template = real_template - np.mean(real_template, axis=0)
real_template = real_template / np.max(real_template, axis=0)


# Initialize data
Ntr = 30
Ns = real_data.shape[0]
data = np.zeros((Ns, Ntr))
tt = np.arange(0, Ns * dt, dt)
order = 3   # 1: linear, 2: quadratic, 3: cubic
s = sinterp.InterpolatedUnivariateSpline(
    tt, real_template, k=order, ext='zeros')

for ntr in range(0, Ntr):
    delay = -10 + 0.1 * ntr
    data[:, ntr] = s(tt - delay)
#     tr = obspy.core.trace.Trace(data=data[:, 0], header=None)
#     st += obspy.core.stream.Stream(traces=[tr])

# counter = 0
# for tr in st:
#     tr.stats.distance = counter
#     tr.stats.sampling_rate = fs
#     counter += 1

# st.plot(type='section')

# Add noise
data += np.random.normal(loc=0, scale=0.1, size=data.shape)


# Plot results
if False:
    options = {
        'shade': True,
        'orientation': 'vertical',
        'color': 'k',
        'scale': None,
        'stretch_factor': 10.,
        'symbols': '-'
    }
    tic = time.time()
    plt.figure()
    sp.wiggle(data=data, tt=tt, type='matplotlib', options=options)
    print(time.time() - tic)
    plt.show()

else:
    tic = time.time()
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    pg.setConfigOptions(antialias=True)  # Enable antialiasing

    app = QtGui.QApplication([])
    options = {
        'shade': True,
        'orientation': 'vertical',
        'color': 'k',
        'scale': None,
        'stretch_factor': 10.,
    }
    win, p = sp.wiggle(data=data, tt=tt, type='pyqtgraph', options=options)

    p.setLabel('left', "Time", units='sec')
    p.setLabel('bottom', "Traces number", units='')
    p.showGrid(x=True, y=True)
    print(time.time() - tic)

    QtGui.QApplication.instance().exec_()
