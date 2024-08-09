import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
from scipy.interpolate import splprep, splev

def load_data(csv_file):
    data = pd.read_csv(csv_file, header=None)
    x_coords = data[2].values
    y_coords = data[3].values
    return np.vstack((x_coords, y_coords)).T

def detect_intersections(curve):
    curve_line = LineString(curve)
    intersections = []
    interrupted_segments = []
    
    for i in range(len(curve) - 2):
        segment_1 = LineString(curve[i:i+2])
        for j in range(i + 2, len(curve) - 1):
            segment_2 = LineString(curve[j:j+2])
            if segment_1.intersects(segment_2):
                intersection_point = segment_1.intersection(segment_2)
                if isinstance(intersection_point, Point):
                    intersections.append(intersection_point)
                    interrupted_segments.append((i, j)) 
    
    return intersections, interrupted_segments

def complete_interrupted_curve(curve, interrupted_segment):
    """Complete the interrupted segment until it intersects another segment."""
    start_idx, end_idx = interrupted_segment
    segment = curve[start_idx:end_idx+1]
    
    direction_vector = segment[-1] - segment[-2]
    
    extended_curve = [segment[-1]]
    for _ in range(10):  
        new_point = extended_curve[-1] + direction_vector * 0.1
        extended_curve.append(new_point)
    
    extended_curve = np.vstack(extended_curve)
    return np.vstack((curve[:end_idx+1], extended_curve))

def main():
    csv_file = 'src/curve_completion/occlusion1.csv'  
    curve = load_data(csv_file)
    
    intersections, interrupted_segments = detect_intersections(curve)
    
    if intersections:
        print("Occlusion detected. Completing the curve...")
        completed_curve = curve.copy()
        
        for interrupted_segment in interrupted_segments:
            completed_curve = complete_interrupted_curve(completed_curve, interrupted_segment)
            break  
        
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
