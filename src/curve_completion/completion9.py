import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from scipy.interpolate import splprep, splev

def load_image(image_path):
    """Load image in grayscale and check if it was loaded successfully."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found or unable to load at {image_path}")
    return image

def detect_edges(image):
    """Use Canny edge detection to find edges in the image."""
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    return edges

def find_contours(edges):
    """Find contours in the edge-detected image."""
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def fit_and_draw_ellipse(contours, image):
    """Fit an ellipse to the largest contour and draw it on the image."""
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) >= 5:  # At least 5 points are required to fit an ellipse
        ellipse = cv2.fitEllipse(largest_contour)
        cv2.ellipse(image, ellipse, (255, 0, 0), 2)
    else:
        ellipse = None
    
    return image, ellipse

def ellipse_to_points(ellipse, num_points=100):
    """Convert the fitted ellipse into a set of points."""
    if ellipse is None:
        return np.empty((0, 2))
    
    center, axes, angle = ellipse
    a, b = axes[0] / 2.0, axes[1] / 2.0
    angle = np.deg2rad(angle)
    
    t = np.linspace(0, 2 * np.pi, num_points)
    X = center[0] + a * np.cos(t) * np.cos(angle) - b * np.sin(t) * np.sin(angle)
    Y = center[1] + a * np.cos(t) * np.sin(angle) + b * np.sin(t) * np.cos(angle)
    
    points = np.vstack((X, Y)).T
    return points

def detect_intersections(curve):
    """Detect intersections in the curve and return intersection points and segments."""
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

def load_csv(csv_path):
    """Load CSV data and extract x, y coordinates."""
    # Read CSV file
    df = pd.read_csv(csv_path, header=None)
    
    # Extract x and y coordinates from the third and fourth columns
    x_coords = df.iloc[:, 2].values
    y_coords = df.iloc[:, 3].values
    
    # Combine x and y coordinates into a single array of points
    points = np.vstack((x_coords, y_coords)).T
    return points

def process_regular_shapes(points):
    """Process the completed curve points to handle regular shapes."""
    # For demonstration, let's just print the points and plot them
    print("Processing Regular Shapes with Points:")
    print(points)
    
    # Here you would implement additional logic to handle regular shapes
    # This might include detecting specific shapes, fitting models, etc.

    # Plotting the processed points
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    ax.plot(points[:, 0], points[:, 1], 'b-', label='Processed Points')
    ax.set_aspect('equal')
    plt.title("Regular Shape Processing")
    plt.legend()
    plt.show()

def main():
    image_path = 'path/to/your/image/Figure_1.png'  # Update this path
    csv_path = 'path/to/your/file.csv'  # Update this path
    
    # Load and process the image
    image = load_image(image_path)
    edges = detect_edges(image)
    contours = find_contours(edges)
    
    if not contours:
        print("No contours found in the image.")
    else:
        result_image, ellipse = fit_and_draw_ellipse(contours, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        
        # Convert ellipse to points
        ellipse_points = ellipse_to_points(ellipse)
        
        # Detect and complete any occlusions in the ellipse points
        intersections, interrupted_segments = detect_intersections(ellipse_points)
        
        completed_curve = ellipse_points.copy()
        for interrupted_segment in interrupted_segments:
            completed_curve = complete_interrupted_curve(completed_curve, interrupted_segment)
        
        # Plot the original and completed curves
        fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
        ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r-', label='Original Ellipse')
        ax.plot(completed_curve[:, 0], completed_curve[:, 1], 'g-', label='Completed Curve')
        ax.set_aspect('equal')
        plt.title("Ellipse Completion")
        plt.legend()
        plt.show()
    
    # Load and process the CSV data
    points = load_csv(csv_path)
    
    if points.size == 0:
        print("No data found in the CSV file.")
    else:
        # Detect and complete any occlusions in the CSV points
        intersections, interrupted_segments = detect_intersections(points)
        
        completed_curve = points.copy()
        for interrupted_segment in interrupted_segments:
            completed_curve = complete_interrupted_curve(completed_curve, interrupted_segment)
        
        # Plot the original and completed curves from CSV
        fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
        ax.plot(points[:, 0], points[:, 1], 'r-', label='Original Points')
        ax.plot(completed_curve[:, 0], completed_curve[:, 1], 'g-', label='Completed Curve')
        ax.set_aspect('equal')
        plt.title("Curve Completion from CSV Data")
        plt.legend()
        plt.show()
        
        # Process the completed curve points for regular shapes
        process_regular_shapes(completed_curve)

if __name__ == "__main__":
    main()
