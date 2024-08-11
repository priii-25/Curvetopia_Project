import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import cv2
import csv

def load_data(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    path_XYs = []
    current_segment = []
    
    for row in data:
        if len(row) >= 2:
            current_segment.append([float(row[-2]), float(row[-1])])
    
    path_XYs = np.array(current_segment)  # Flat array of points
    return path_XYs

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
    start_idx, end_idx = interrupted_segment
    segment = curve[start_idx:end_idx+1]
    
    direction_vector = segment[-1] - segment[-2]
    
    extended_curve = [segment[-1]]
    for _ in range(10):  
        new_point = extended_curve[-1] + direction_vector * 0.1
        extended_curve.append(new_point)
    
    extended_curve = np.vstack(extended_curve)
    return np.vstack((curve[:end_idx+1], extended_curve))

def fit_and_draw_ellipse(curve_points):
    curve_points = curve_points.astype(np.float32)  
    
    if len(curve_points) >= 5: 
        ellipse = cv2.fitEllipse(curve_points)
        
        ellipse_points = ellipse_to_points(ellipse)
        return ellipse_points, ellipse
    else:
        print("Not enough points to fit an ellipse.")
        return curve_points, None

def ellipse_to_points(ellipse, num_points=100):
    center, axes, angle = ellipse
    a, b = axes[0] / 2.0, axes[1] / 2.0
    angle = np.deg2rad(angle)
    
    t = np.linspace(0, 2 * np.pi, num_points)
    X = center[0] + a * np.cos(t) * np.cos(angle) - b * np.sin(t) * np.sin(angle)
    Y = center[1] + a * np.cos(t) * np.sin(angle) + b * np.sin(t) * np.cos(angle)
    
    points = np.vstack((X, Y)).T
    return points

def main():
    csv_file = 'src\curve_completion\occlusion1.csv'  
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
    
    ellipse_points, fitted_ellipse = fit_and_draw_ellipse(completed_curve)
    
    plt.figure()
    
    plt.scatter(curve[:, 0], curve[:, 1], c='blue', label='Original Points', marker='o')
    
    plt.scatter(completed_curve[:, 0], completed_curve[:, 1], c='green', label='Completed Points', marker='x')
    
    if fitted_ellipse is not None:
        plt.scatter(ellipse_points[:, 0], ellipse_points[:, 1], c='red', label='Ellipse Points', marker='^')
    
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Curve Analysis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    main()