# test_python_spectrogram.py
import numpy as np
from scipy import signal
import timeit
import time


if __name__ == '__main__':
    # Construct data
    fs = 10e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.001 * fs / 2
    tt = np.arange(N) / fs
    freq = np.linspace(1e3, 2e3, N)
    x = amp * np.sin(2 * np.pi * freq * tt)
    x += np.random.normal(scale=np.sqrt(noise_power), size=tt.shape)

    tic = time.time()
    f, t, Sxx = signal.spectrogram(x, fs)
    toc = time.time() - tic

    # Time it
    t_scipy = timeit.Timer(lambda: signal.spectrogram(x, fs))

    # Print results
    print("Testing spectrogram speed:")
    print("\tMatlab benchmark is {:.5e} sec".format(0.004628029410500))
    print("\tSingle line execution time is {:.5e} sec".format(toc))
    print("\tScipy.singal.spectrogram costs {:.5e} sec".format(
        min(t_scipy.repeat(repeat=3, number=1))))

    # Plot spectrogram
    # plt.pcolormesh(t, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
