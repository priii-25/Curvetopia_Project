import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from scipy.interpolate import splprep, splev

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

def plot(path_XYs, title=""):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()

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

def estimate_occluded_curve(start_point, end_point, num_points=10):
    """Estimate the occluded curve between start_point and end_point with additional intermediate points."""
    x_vals = np.linspace(start_point.x, end_point.x, num_points)
    y_vals = np.linspace(start_point.y, end_point.y, num_points)
    tck, _ = splprep([x_vals, y_vals], s=0, k=3)
    u_new = np.linspace(0, 1, 100)
    x_new, y_new = splev(u_new, tck)
    return np.vstack((x_new, y_new)).T

def complete_interrupted_curve(curve, interrupted_segment):
    """Complete the interrupted segment until it intersects another segment."""
    start_idx, end_idx = interrupted_segment
    
    # Ensure indices are within bounds
    if start_idx >= len(curve) or end_idx >= len(curve):
        print(f"Index out of bounds: start_idx={start_idx}, end_idx={end_idx}, curve length={len(curve)}")
        return curve
    
    start_point = Point(curve[start_idx])
    end_point = Point(curve[end_idx])
    print(f"Completing curve between points {start_point} and {end_point}")
    estimated_curve = estimate_occluded_curve(start_point, end_point)
    
    # Merge the estimated curve back into the original curve
    completed_curve = np.vstack((curve[:start_idx+1], estimated_curve, curve[end_idx+1:]))
    return completed_curve

def main():
    csv_file = 'src/curve_completion/occlusion1.csv'  
    path_XYs = read_csv(csv_file)
    
    completed_paths = []

    for path in path_XYs:
        for curve in path:
            intersections, interrupted_segments = detect_intersections(curve)
    
            if intersections:
                print("Occlusion detected. Completing the curve...")
                completed_curve = curve.copy()
                
                for interrupted_segment in interrupted_segments:
                    completed_curve = complete_interrupted_curve(completed_curve, interrupted_segment)
                    
                    # Recalculate intersections after completing a segment
                    intersections, interrupted_segments = detect_intersections(completed_curve)
                
            else:
                completed_curve = curve

            completed_paths.append([completed_curve])  # Add the completed curve to the paths

    plot(completed_paths, title="Occlusion Detection and Curve Completion")

if __name__ == "__main__":
    main()
