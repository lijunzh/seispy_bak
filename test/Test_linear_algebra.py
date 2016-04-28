# Test common linear algebra speed
import numpy as np
from scipy import linalg
import timeit


if __name__ == '__main__':
    # Generate random matrix (normal distribution)
    size = 1000

    # Beginning remark:
    print("Speed comparison between MATLAB(w/ MKL) and Numpy/Scipy(w/o MKL)\n")

    t_np_randn = timeit.Timer(lambda: np.random.randn(size, size))
    t_np_normal = timeit.Timer(
        lambda: np.random.normal(scale=1.0, size=(size, size)))
    print("Test Normal distribution generation speed:")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(.0143))
    print("\tNumpy.random.randn:\t\t{:.5e} sec".format(
        min(t_np_randn.repeat(repeat=3, number=1))))
    print("\tNumpy.random.normal:\t{:.5e} sec".format(
        min(t_np_normal.repeat(repeat=3, number=1))))

    # Matrix summation
    size = 1000
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)

    t_sum = timeit.Timer(lambda: A + B)
    print("Test matrix summation:")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(0.00060394))
    print("\tNumpy.array:\t\t\t{:.5e} sec".format(
        min(t_sum.repeat(repeat=3, number=1))))

    # Matrix inverse
    size = 1000
    A = np.random.randn(size, size)

    t_inverse = timeit.Timer(lambda: linalg.inv(A))
    print("Test matrix inverse:")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(0.0258))
    print("\tNumpy inverse:\t\t\t{:.5e} sec".format(
        min(t_inverse.repeat(repeat=3, number=1))))

    # Matrix transpose
    size = 1000
    A = np.random.randn(size, size)

    t_transpose = timeit.Timer(lambda: A.T)
    print("Test matrix transpose:")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(0.0015))
    print("\tNumpy transpose:\t\t{:.5e} sec".format(
        min(t_transpose.repeat(repeat=3, number=1))))

    # Matrix element multiplication
    size = 1000
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)

    t_em = timeit.Timer(lambda: A * B)
    print("Test matrix element multiplication:")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(0.0005991))
    print("\tNumpy.array:\t\t\t{:.5e} sec".format(
        min(t_em.repeat(repeat=3, number=1))))

    # Matrix multiplication
    size = 1000
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)

    t_em = timeit.Timer(lambda: A.dot(B))
    print("Test matrix multiplication:")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(0.0134))
    print("\tNumpy.array:\t\t\t{:.5e} sec".format(
        min(t_em.repeat(repeat=3, number=1))))

    # Solve linear system
    size = 1000
    A = np.random.randn(size, size)
    b = np.random.randn(size, 1)
    t_solve = timeit.Timer(lambda: np.linalg.solve(A, b))
    print("Test solving lienar system:")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(0.0141))
    print("\tNumpy.array:\t\t\t{:.5e} sec".format(
        min(t_em.repeat(repeat=3, number=1))))

    # Find determinant
    size = 1000
    A = np.random.randn(size, size)
    t_solve = timeit.Timer(lambda: linalg.det(A))
    print("Test finding determinant:")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(0.0086))
    print("\tNumpy.array:\t\t\t{:.5e} sec".format(
        min(t_em.repeat(repeat=3, number=1))))

    # Compute norm
    size = 1000
    A = np.random.randn(size, size)
    t_solve = timeit.Timer(lambda: linalg.norm(A, 'fro'))
    print("Test computing norm:")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(0.0046))
    print("\tNumpy.array:\t\t\t{:.5e} sec".format(
        min(t_em.repeat(repeat=3, number=1))))

    # Solve least-square
    c1, c2 = 5.0, 2.0
    i = np.r_[1:1000]
    xi = 0.1 * i
    yi = c1 * np.exp(-xi) + c2 * xi
    zi = yi + 0.05 * np.max(yi) * np.random.randn(len(yi))

    A = np.c_[np.exp(-xi)[:, np.newaxis], xi[:, np.newaxis]]
    t_ls = timeit.Timer(lambda: linalg.lstsq(A, zi))
    print("Test least-squares:")
    print("\tMatlab benchmark(?):\t{:.5e} sec".format(0.0065))
    print("\tscipy.linalg.lstsq:\t\t{:.5e} sec".format(
        min(t_ls.repeat(repeat=3, number=1))))

    # Eigenvalue decomposition
    size = 1000
    A = np.random.randn(size, size)
    t_eig1 = timeit.Timer(lambda: linalg.eig(A))
    t_eig2 = timeit.Timer(lambda: linalg.eigvals(A))
    print("Test eigenvalue decomposition(full):")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(1.286))
    print("\tNumpy.array:\t\t\t{:.5e} sec".format(
        min(t_eig1.repeat(repeat=3, number=1))))
    print("Test eigenvalue decomposition(value):")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(0.6693))
    print("\tNumpy.array:\t\t\t{:.5e} sec".format(
        min(t_eig2.repeat(repeat=3, number=1))))

    # Singular value decomposition
    size = 1000
    A = np.random.randn(size, size)
    t_svd = timeit.Timer(lambda: linalg.svd(A))
    print("Test singular value decomposition:")
    print("\tMatlab benchmark:\t\t{:.5e} sec".format(0.3638))
    print("\tNumpy.array:\t\t\t{:.5e} sec".format(
        min(t_svd.repeat(repeat=3, number=1))))
