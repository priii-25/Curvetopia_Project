import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from scipy.interpolate import interp1d

def load_data(csv_file):
    data = pd.read_csv(csv_file, header=None)
    x_coords = data[2].values
    y_coords = data[3].values
    return x_coords, y_coords

def detect_interruption(curve1, curve2):
    line1 = LineString(curve1)
    line2 = LineString(curve2)
    intersection = line1.intersection(line2)
    return not intersection.is_empty

def find_interruptions(curve, gap_threshold=10.0):
    interruptions = []
    dists = np.sqrt(np.diff(curve[:, 0])**2 + np.diff(curve[:, 1])**2)
    large_gaps = np.where(dists > gap_threshold)[0]

    if len(large_gaps) > 0:
        start = 0
        for gap in large_gaps:
            end = gap + 1
            interruptions.append((start, end))
            start = end
        interruptions.append((start, len(curve)))
    else:
        interruptions.append((0, len(curve)))
    
    return interruptions

def interpolate_missing_points(curve, points_before, points_after):
    if len(points_before) == 0 or len(points_after) == 0:
        return np.empty((0, 2))

    x_curve = curve[:, 0]
    y_curve = curve[:, 1]

    idx_before = len(points_before) - 1
    idx_after = 0
    
    interp_x = interp1d([idx_before, idx_after], [points_before[-1][0], points_after[0][0]], fill_value="extrapolate")
    interp_y = interp1d([idx_before, idx_after], [points_before[-1][1], points_after[0][1]], fill_value="extrapolate")

    x_interp = interp_x(np.arange(idx_before + 1, idx_after))
    y_interp = interp_y(np.arange(idx_before + 1, idx_after))
    
    interpolated_points = np.vstack((x_interp, y_interp)).T
    return interpolated_points

def complete_curve(curve, interruptions):
    completed_curve = []
    prev_index = 0

    for interruption in interruptions:
        start, end = interruption
        points_before = curve[prev_index:start]
        points_after = curve[end:]
        
        completed_curve.extend(points_before)

        if len(points_before) > 0 and len(points_after) > 0:
            interpolated_points = interpolate_missing_points(curve[start:end], points_before, points_after)
            completed_curve.extend(interpolated_points)
        
        prev_index = end

    completed_curve.extend(curve[prev_index:])
    return np.array(completed_curve)

def main():
    csv_file = 'src\curve_completion\occlusion1.csv'
    x_coords, y_coords = load_data(csv_file)
    curve = np.vstack((x_coords, y_coords)).T

    interruptions = find_interruptions(curve, gap_threshold=10.0)
    completed_curve = complete_curve(curve, interruptions)

    if completed_curve.ndim == 1:
        completed_curve = completed_curve.reshape(-1, 2)

    plt.plot(curve[:, 0], curve[:, 1], 'bo-', label='Original Curve')
    plt.plot(completed_curve[:, 0], completed_curve[:, 1], 'r-', label='Completed Curve')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Curve Interruption and Completion')
    plt.show()

if __name__ == "__main__":
    main()
