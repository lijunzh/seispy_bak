import numpy as np
data = np.random.normal(loc=0., scale=1., size=100)
# Normalize data
data = data / np.std(data, axis=0)






import scipy.optimize as optimize

rosen = lambda x: sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0
                      + (1 - x[:-1])**2.0)
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = optimize.minimize(rosen, x0, method='BFGS',
                        options={'disp': True})



def classify(values, boundary=0):
    '''Classifies values as being below 
    (False) or above (True) a boundary.
    '''
    return [(True if v > boundary else False)
            for v in values]

# Call the above function
my_values = np.array([1])
classify(my_values, boundary=0.5)

A = np.random.randn(3, 3)
b = np.random.rand(3)
x = np.linalg.solve(A, b)
