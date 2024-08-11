import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import cv2

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
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

def fit_ellipse_through_segment(segment):
    if len(segment) >= 5:
        ellipse = cv2.fitEllipse(segment.astype(np.float32))
        return ellipse
    return None

def ellipse_to_points(ellipse, num_points=100):
    center, axes, angle = ellipse
    a, b = axes[0] / 2.0, axes[1] / 2.0
    angle = np.deg2rad(angle)
    
    t = np.linspace(0, 2 * np.pi, num_points)
    X = center[0] + a * np.cos(t) * np.cos(angle) - b * np.sin(t) * np.sin(angle)
    Y = center[1] + a * np.cos(t) * np.sin(angle) + b * np.sin(t) * np.cos(angle)
    
    points = np.vstack((X, Y)).T
    return points

def plot(original_curves, fitted_ellipses, title=""):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    
    # Plot original curves
    for curve in original_curves:
        ax.plot(curve[:, 0], curve[:, 1], 'bo-', linewidth=2, markersize=2, label='Original Curve')
    
    # Plot fitted ellipses
    for ellipse in fitted_ellipses:
        ax.plot(ellipse[:, 0], ellipse[:, 1], 'r-', linewidth=2, label='Fitted Ellipse')
    
    ax.set_aspect('equal')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    csv_file = 'src/curve_completion/occlusion1.csv'
    path_XYs = read_csv(csv_file)
    
    all_original_curves = []
    all_fitted_ellipses = []
    
    for XYs in path_XYs:
        for curve in XYs:
            all_original_curves.append(curve)
            intersections, interrupted_segments = detect_intersections(curve)
            
            if intersections:
                print("Occlusion detected. Completing the curve...")
                completed_curve = curve.copy()
                
                for interrupted_segment in interrupted_segments:
                    completed_curve = complete_interrupted_curve(completed_curve, interrupted_segment)
                
                ellipses = []
                for interrupted_segment in interrupted_segments:
                    segment = completed_curve[interrupted_segment[0]:interrupted_segment[1]+1]
                    ellipse = fit_ellipse_through_segment(segment)
                    
                    if ellipse is not None:
                        ellipse_points = ellipse_to_points(ellipse)
                        ellipses.append(ellipse_points)
                
                all_fitted_ellipses.extend(ellipses)
            else:
                ellipse = fit_ellipse_through_segment(curve)
                if ellipse is not None:
                    all_fitted_ellipses.append(ellipse_to_points(ellipse))
    
    plot(all_original_curves, all_fitted_ellipses, title="Occlusion Detection and Ellipse Regularization")


if __name__ == "__main__":
    main()
