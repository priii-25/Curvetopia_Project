import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import pandas as pd

def complete_shape_with_bspline(csv_file):
    data = pd.read_csv(csv_file, header=None)
    
    x = data.iloc[:, 2].values
    y = data.iloc[:, 3].values

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    try:
        tck, u = spi.splprep([x, y], s=0)
    except ValueError as e:
        raise ValueError(f"Error in B-spline fitting: {e}")

    unew = np.linspace(0, 1, 100)
    out = spi.splev(unew, tck)

    plt.figure()
    plt.plot(x, y, 'ro', label='Original Points')
    plt.plot(out[0], out[1], 'b-', label='B-Spline Fit')
    plt.legend()
    plt.title('Shape Completion using B-Spline')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    completed_shape = np.vstack((out[0], out[1])).T
    np.savetxt("completed_shape_bspline.csv", completed_shape, delimiter=",", header="x,y", comments="")

complete_shape_with_bspline('src\curve_completion\occlusion1.csv')
