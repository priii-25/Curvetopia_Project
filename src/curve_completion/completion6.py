import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splprep, splev

def load_data(csv_file):
    data = pd.read_csv(csv_file, header=None)
    x_coords = data[2].values
    y_coords = data[3].values
    return np.vstack((x_coords, y_coords)).T

def detect_half_shape(curve):
    """Detect if half of the shape is formed and the rest is occluded."""
    length = curve.shape[0]
    if length >= 5: 
        return True
    return False

def b_spline_interpolation(control_points, knot_vector, degree):
    """Interpolate a B-spline curve using control points and knot vector."""
    spl = BSpline(knot_vector, control_points, degree)
    sample_points = np.linspace(0, 1, 100)
    interpolated_points = spl(sample_points)
    return interpolated_points

def complete_half_shape(curve):
    """Complete the shape if half is detected."""
    control_points = curve  

    degree = 3
    knot_vector = np.linspace(0, 1, len(control_points) + degree + 1)

    # Interpolate the curve
    interpolated_curve = b_spline_interpolation(control_points, knot_vector, degree)
    
    return interpolated_curve

def main():
    csv_file = 'src\curve_completion\occlusion1.csv'  
    curve = load_data(csv_file)
    
    if detect_half_shape(curve):
        print("Half of the shape detected. Completing the shape...")
        completed_curve = complete_half_shape(curve)
    else:
        completed_curve = curve

    plt.plot(curve[:, 0], curve[:, 1], 'bo-', label='Original Curve')
    plt.plot(completed_curve[:, 0], completed_curve[:, 1], 'r-', label='Completed Curve')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Shape Detection and Completion')
    plt.show()

if __name__ == "__main__":
    main()
