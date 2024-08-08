import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splprep, splev

def detect_occlusion(x, y, threshold=10):
    """Detect occlusion by identifying large gaps between points."""
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    occlusion_indices = np.where(dists > threshold)[0]
    return occlusion_indices

def generate_control_points(x, y, occlusion_indices):
    """Generate control points from the visible points, ignoring occlusions."""
    if len(occlusion_indices) > 0:
        segments = np.split(np.arange(len(x)), occlusion_indices + 1)
        control_points = np.array([[x[seg].mean(), y[seg].mean()] for seg in segments])
    else:
        control_points = np.array([x, y]).T
    return control_points

def fit_b_spline(control_points, degree=3):
    """Fit a B-spline using the control points."""
    n = len(control_points) - 1
    k = degree

    # Check if there are enough control points
    if n < (degree + 1):
        raise ValueError(f"Insufficient number of control points for B-spline of degree {degree}. Need at least {degree + 1} control points.")
    
    # Create a uniform knot vector
    t = np.linspace(0, 1, n + k + 1)  # Uniform knot vector
    t = np.clip(t, 0, 1)
    
    x = control_points[:, 0]
    y = control_points[:, 1]
    
    return t, BSpline(t, np.vstack([x, y]).T, k)




def evaluate_b_spline(spline, num_points=100):
    """Evaluate the B-spline at a given number of points."""
    u = np.linspace(0, 1, num_points)
    points = spline(u)
    return points[:, 0], points[:, 1]

def main():
    csv_file = 'src/curve_completion/occlusion2.csv'  
    data = pd.read_csv(csv_file, header=None)

    x_coords = data[2].values
    y_coords = data[3].values

    occlusion_indices = detect_occlusion(x_coords, y_coords)

    control_points = generate_control_points(x_coords, y_coords, occlusion_indices)

    if len(control_points) < 8:
        print(f"Warning: Only {len(control_points)} control points generated, which is insufficient for a B-spline of degree 3.")
        if len(control_points) < 2:
            print("Too few control points for any B-spline or interpolation. Cannot proceed.")
            return
        
        # Use linear interpolation as a fallback
        x_spline, y_spline = x_coords, y_coords
        label = 'Linear Interpolation'
    else:
        degree = 3
        t, spline = fit_b_spline(control_points, degree)
        x_spline, y_spline = evaluate_b_spline(spline, num_points=400)
        label = 'Fitted B-Spline'

    fig, ax = plt.subplots()
    ax.plot(x_coords, y_coords, 'bo', label='Visible Points')
    ax.plot(control_points[:, 0], control_points[:, 1], 'ro', label='Control Points')
    ax.plot(x_spline, y_spline, 'g-', label=label)

    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Occluded Shape Detection and Completion')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == "__main__":
    main()