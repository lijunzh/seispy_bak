import numpy
import scipy.fftpack as fft
#import pyfftw
import timeit
import time


# def generate_fftw(size):
#     a = pyfftw.empty_aligned(size, dtype='complex128')
#     b = pyfftw.empty_aligned(size, dtype='complex128')

#     # forward
#     fft_object = pyfftw.FFTW(a, b)
#     # backward
#     # c = pyfftw.empty_aligned(size, dtype='complex128')
#     # ifft_object = pyfftw.FFTW(b, c, direction='FFTW_BACKWARD')
#     return fft_object, a, b


# def builders_fftw(size):
#     a = pyfftw.empty_aligned(size, dtype='complex128')

#     # forward
#     fft_object = pyfftw.builders.fft(a)

#     return fft_object, a

if __name__ == '__main__':
    size = 2 ** 10

    print(
        "Python speed test of FFT function using {} length complex vector\n".format(size))

    ar, ai = numpy.random.randn(2, size)
    test_data = ar + 1j * ai

    # Numpy benchmark
    t_np = timeit.Timer(lambda: numpy.fft.fft(test_data))
    print("Numpy benchmark:\t {:.5e} sec".format(
        min(t_np.repeat(repeat=3, number=1))))

    # Scipy benchmark
    t_scipy = timeit.Timer(lambda: fft.fft(test_data))
    print("Scipy benchmark:\t {:.5e} sec".format(
        min(t_scipy.repeat(repeat=3, number=1))))

    # MATLAB:
    print("MATLAB benchmark:\t 9.75997e-06 sec")
    print('')
    # #------------ pyfftw.FFTW ------------------
    # fft_object, a, b = generate_fftw(size)

    # # Generate some data
    # a[:] = test_data

    # # Test single call
    # tic = time.time()
    # a_fft = fft_object()
    # toc = time.time() - tic

    # # testit results
    # t_prep = timeit.Timer(lambda: generate_fftw(size))
    # t_exec = timeit.Timer(lambda: fft_object())

    # Print results
    # print("Test FFTW using pyfftw.FFTW:")
    # print("\tSingle call of fft_object():\t\t{:.5e} sec".format(toc))
    # print("\tThe preparation time of FFTW:\t\t {:.5e} sec".format(
    #     min(t_prep.repeat(repeat=3, number=1))))
    # print("\tThe execute time of FFTW:\t\t {:.5e} sec".format(
    #     min(t_exec.repeat(repeat=3, number=1))))
    # print(
    #     "\tThe output of fft_object is the same as the object variable:",
    #     a_fft is b)

    # print('')
    # # ------------ pyfftw.builders.fft ------------------
    # fft_object, a = builders_fftw(size)

    # # Generate some data
    # a[:] = test_data

    # # Test single call
    # tic = time.time()
    # a_fft = fft_object()
    # toc = time.time() - tic

    # # testit results
    # t_prep = timeit.Timer(lambda: generate_fftw(size))
    # t_exec = timeit.Timer(lambda: fft_object())

    # # Print results
    # print("Test FFTW using pyfftw.builders:")
    # print("\tSingle call of fft_object():\t\t{:.5e} sec".format(toc))
    # print("\tThe preparation time of FFTW:\t\t {:.5e} sec".format(
    #     min(t_prep.repeat(repeat=3, number=1))))
    # print("\tThe execute time of FFTW:\t\t{:.5e} sec".format(
    #     min(t_exec.repeat(repeat=3, number=1))))
    # print("\tThe output of fft_object is the same as the object variable:",
    #       a_fft is fft_object())
