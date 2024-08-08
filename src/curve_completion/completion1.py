import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def load_csv(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    x_coords = data[:, 2]
    y_coords = data[:, 3]
    return x_coords, y_coords

def plot_shape(x, y, title="Incomplete Shape"):
    plt.figure()
    plt.plot(x, y, 'o', label='Original Points')
    plt.title(title)
    plt.legend()
    plt.show()

def remove_invalid_values(x, y):
    valid_indices = ~np.isnan(x) & ~np.isnan(y) & np.isfinite(x) & np.isfinite(y)
    return x[valid_indices], y[valid_indices]

def bspline_interpolation(x, y, s=0):
    x, y = remove_invalid_values(x, y)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Not enough valid points for interpolation")
    
    tck, u = splprep([x, y], s=s)
    u_new = np.linspace(u.min(), u.max(), len(x) * 10)
    x_new, y_new = splev(u_new, tck, der=0)
    return x_new, y_new

def plot_completed_shape(x, y, x_new, y_new, title="Completed Shape"):
    plt.figure()
    plt.plot(x, y, 'o', label='Original Points')
    plt.plot(x_new, y_new, '-', label='Completed Shape')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    file_path = 'src\curve_completion\occlusion1.csv'  
    x, y = load_csv(file_path)
    x, y = remove_invalid_values(x, y)
    
    plot_shape(x, y, "Incomplete Shape")
    x_new, y_new = bspline_interpolation(x, y, s=2)
    plot_completed_shape(x, y, x_new, y_new, "Completed Shape")


if __name__ == "__main__":
    main()
