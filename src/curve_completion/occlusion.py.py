import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import cv2
import csv
from svgwrite import Drawing

def load_data(csv_path):
    """
    Load XY curve data from a CSV file.

    Parameters:
    csv_path (str): The path to the CSV file containing the curve data.

    Returns:
    np.ndarray: A numpy array containing the XY coordinates of the curve.
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    path_XYs = []
    current_segment = []
    
    for row in data:
        if len(row) >= 2:
            current_segment.append([float(row[-2]), float(row[-1])])
    
    path_XYs = np.array(current_segment)
    return path_XYs

def detect_intersections(curve):
    """
    Detect intersections or occlusions in the curve by checking for overlapping line segments.

    Parameters:
    curve (np.ndarray): A numpy array of XY coordinates representing the curve.

    Returns:
    list: A list of intersection points.
    list: A list of tuples representing the indices of the interrupted segments.
    """
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
    """
    Complete an interrupted segment of the curve by extending it.

    Parameters:
    curve (np.ndarray): The original curve as an array of XY points.
    interrupted_segment (tuple): The start and end indices of the interrupted segment.

    Returns:
    np.ndarray: The extended curve with the interrupted segment completed.
    """
    start_idx, end_idx = interrupted_segment
    segment = curve[start_idx:end_idx+1]
    
    direction_vector = segment[-1] - segment[-2]  
    
    extended_curve = [segment[-1]]
    for _ in range(10):  
        new_point = extended_curve[-1] + direction_vector * 0.1
        extended_curve.append(new_point)
    
    extended_curve = np.vstack(extended_curve)
    return np.vstack((curve[:end_idx+1], extended_curve))

def detect_shape_type(curve_points):
    """
    Determine the type of shape the curve points resemble (ellipse, circle, or line).

    Parameters:
    curve_points (np.ndarray): The array of points representing the curve.

    Returns:
    str: The detected shape type ('ellipse', 'circle', 'line', or None).
    """
    curve_points = curve_points.astype(np.float32)

    if len(curve_points) >= 5:
        ellipse = cv2.fitEllipse(curve_points)
        aspect_ratio = min(ellipse[1]) / max(ellipse[1]) 

        if aspect_ratio < 0.9:
            return 'ellipse'
        else:
            return 'circle'
    
    elif len(curve_points) >= 2:
        return 'line'
    
    return None

def fit_shape(curve_points, shape_type):
    """
    Fit a geometric shape (ellipse, circle, or line) to the curve points.

    Parameters:
    curve_points (np.ndarray): The array of points representing the curve.
    shape_type (str): The type of shape to fit ('ellipse', 'circle', or 'line').

    Returns:
    np.ndarray: The points of the fitted shape.
    str: The type of shape fitted.
    """
    curve_points = curve_points.astype(np.float32)  
    
    if shape_type == 'ellipse' and len(curve_points) >= 5:
        ellipse = cv2.fitEllipse(curve_points)
        ellipse_points = ellipse_to_points(ellipse)
        return ellipse_points, 'ellipse'
    
    elif shape_type == 'circle' and len(curve_points) >= 3:
        (x, y), radius = cv2.minEnclosingCircle(curve_points)
        circle_points = circle_to_points((x, y), radius)
        return circle_points, 'circle'
    
    elif shape_type == 'line' and len(curve_points) >= 2:
        [vx, vy, x, y] = cv2.fitLine(curve_points, cv2.DIST_L2, 0, 0.01, 0.01)
        line_points = line_to_points((x, y), (vx, vy))
        return line_points, 'line'
    
    else:
        print(f"Not enough points to fit a {shape_type}.")
        return curve_points, None

def ellipse_to_points(ellipse, num_points=100):
    """
    Convert an ellipse representation to a set of points.

    Parameters:
    ellipse (tuple): The ellipse parameters (center, axes, angle).
    num_points (int): Number of points to generate along the ellipse.

    Returns:
    np.ndarray: An array of points representing the ellipse.
    """
    center, axes, angle = ellipse
    a, b = axes[0] / 2.0, axes[1] / 2.0
    angle = np.deg2rad(angle)
    
    t = np.linspace(0, 2 * np.pi, num_points)
    X = center[0] + a * np.cos(t) * np.cos(angle) - b * np.sin(t) * np.sin(angle)
    Y = center[1] + a * np.cos(t) * np.sin(angle) + b * np.sin(t) * np.cos(angle)
    
    points = np.vstack((X, Y)).T
    return points

def circle_to_points(center, radius, num_points=100):
    """
    Convert a circle representation to a set of points.

    Parameters:
    center (tuple): The center of the circle.
    radius (float): The radius of the circle.
    num_points (int): Number of points to generate along the circle.

    Returns:
    np.ndarray: An array of points representing the circle.
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    X = center[0] + radius * np.cos(t)
    Y = center[1] + radius * np.sin(t)
    
    points = np.vstack((X, Y)).T
    return points

def line_to_points(point, direction, length=100):
    """
    Convert a line representation to a set of points.

    Parameters:
    point (tuple): A point on the line.
    direction (tuple): The direction vector of the line.
    length (int): Length to extend the line in both directions.

    Returns:
    np.ndarray: An array of points representing the line.
    """
    t = np.linspace(-length, length, 100)
    X = point[0] + t * direction[0]
    Y = point[1] + t * direction[1]
    
    points = np.vstack((X, Y)).T
    return points

def save_as_svg(original_curve, fitted_curve, output_file):
    """
    Save the original and fitted curves as an SVG file.

    Parameters:
    original_curve (np.ndarray): The points of the original curve.
    fitted_curve (np.ndarray): The points of the fitted curve.
    output_file (str): The output file path for the SVG.
    """
    dwg = Drawing(output_file, profile='tiny')
    
    for point in original_curve:
        dwg.add(dwg.circle(center=point, r=1, fill='blue'))  # Original curve in blue
    for point in fitted_curve:
        dwg.add(dwg.circle(center=point, r=1, fill='red'))  # Fitted curve in red
    
    dwg.save()
    print(f"Saved SVG with points to {output_file}")

def save_as_csv(original_curve, fitted_curve, output_file):
    """
    Save the original and fitted curves as a CSV file.

    Parameters:
    original_curve (np.ndarray): The points of the original curve.
    fitted_curve (np.ndarray): The points of the fitted curve.
    output_file (str): The output file path for the CSV.
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Original Curve'])
        for point in original_curve:
            writer.writerow([0.0, 0.0, point[0], point[1]])  # Write original points
            
        writer.writerow(['Fitted Curve'])
        for point in fitted_curve:
            writer.writerow([0.0, 0.0, point[0], point[1]])  # Write fitted points
            
    print(f"Saved CSV to {output_file}")

def main():
    """
    Main function to load the curve, detect and complete occlusions, fit a shape, and save results.
    """
    csv_file = 'src/curve_completion/occlusion1.csv'  
    curve = load_data(csv_file)
    
    intersections, interrupted_segments = detect_intersections(curve)
    
    if intersections:
        print("Occlusion detected. Completing the curve...")
        completed_curve = curve.copy()
        
        for interrupted_segment in interrupted_segments:
            completed_curve = complete_interrupted_curve(completed_curve, interrupted_segment)

    else:
        completed_curve = curve
    
    shape_type = detect_shape_type(completed_curve) 
    
    if shape_type:
        print(f"Detected shape type: {shape_type}")
        shape_points, fitted_shape = fit_shape(completed_curve, shape_type)
    else:
        print("Could not detect a valid shape type. Returning the original curve.")
        shape_points, fitted_shape = completed_curve, None
    
    # Plot the original and fitted curves
    plt.figure()
    plt.scatter(curve[:, 0], curve[:, 1], c='blue', label='Original Points', marker='o')
    if fitted_shape is not None:
        plt.scatter(shape_points[:, 0], shape_points[:, 1], c='red', label=f'{fitted_shape.capitalize()} Points', marker='o')
    
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Curve Analysis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    # Save the output as SVG and CSV
    output_svg = 'output_combined.svg'
    output_csv = 'output_combined.csv'
    
    save_as_svg(curve, shape_points, output_svg)
    save_as_csv(curve, shape_points, output_csv)

if __name__ == "__main__":
    main()
