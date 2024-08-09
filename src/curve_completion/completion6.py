import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import split
from scipy.interpolate import splprep, splev

def load_data(csv_file):
    data = pd.read_csv(csv_file, header=None)
    x_coords = data[2].values
    y_coords = data[3].values
    return np.vstack((x_coords, y_coords)).T

def detect_intersections(curve):
    """Detect intersections within the curve which might indicate occlusions."""
    curve_line = LineString(curve)
    intersections = []
    
    for i in range(len(curve) - 2):
        segment_1 = LineString(curve[i:i+2])
        for j in range(i + 2, len(curve) - 1):
            segment_2 = LineString(curve[j:j+2])
            if segment_1.intersects(segment_2):
                intersection_point = segment_1.intersection(segment_2)
                if isinstance(intersection_point, Point):
                    intersections.append(intersection_point)
    
    return intersections

def complete_curve_around_intersections(curve, intersections):
    """Complete the curve around the detected intersection points."""
    completed_curve = curve.copy()

    # Identify control points around intersections and fit a curve
    for intersection in intersections:
        intersection_coords = np.array(intersection.coords[0])
        idx = np.argmin(np.sum((curve - intersection_coords) ** 2, axis=1))
        
        # Select a segment of the curve around the intersection
        segment = curve[max(0, idx-5):min(len(curve), idx+6)]
        
        # Interpolate a curve around the segment to 'complete' it
        tck, _ = splprep([segment[:, 0], segment[:, 1]], s=0)
        u_new = np.linspace(0, 1, 100)
        x_new, y_new = splev(u_new, tck)
        interpolated_segment = np.vstack((x_new, y_new)).T
        
        # Replace the old segment with the interpolated one
        completed_curve = np.vstack((completed_curve[:max(0, idx-5)], interpolated_segment, completed_curve[min(len(curve), idx+6):]))
    
    return completed_curve

def main():
    csv_file = 'src/curve_completion/occlusion1.csv'  
    curve = load_data(csv_file)
    
    # Detect intersections in the curve
    intersections = detect_intersections(curve)
    
    if intersections:
        print("Occlusion detected. Completing the curve...")
        completed_curve = complete_curve_around_intersections(curve, intersections)
    else:
        completed_curve = curve
    
    plt.plot(curve[:, 0], curve[:, 1], 'bo-', label='Original Curve')
    plt.plot(completed_curve[:, 0], completed_curve[:, 1], 'r-', label='Completed Curve')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Occlusion Detection and Curve Completion')
    plt.show()

if __name__ == "__main__":
    main()
