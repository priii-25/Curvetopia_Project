import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import svgwrite
import cairosvg
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import corner_harris
import cv2

# Your provided function (CANNOT BE EDITED)
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

def plot(path_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()

def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    colours = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))
    dwg.add(group)
    dwg.save()
    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact * W, output_height=fact * H, background_color='white')

# Shape Regularization Algorithms

def detect_line(points):
    """Detects straight lines using linear regression."""
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression().fit(X, y)
    score = model.score(X, y)
    threshold = 0.95 
    if score > threshold:
        return model.coef_[0], model.intercept_ 
    else:
        return None

def detect_circle_kasa(points, max_iterations=100, tolerance=1e-6):
    """Detects circles using Kasa's method."""
    n = len(points)
    if n < 3:  # Need at least 3 points to fit a circle
        return None

    # 1. Calculate centroid (initial circle center)
    centroid = np.mean(points, axis=0)
    xc, yc = centroid

    # 2. Iterative optimization
    for _ in range(max_iterations):
        # Calculate distances from centroid to points
        distances = np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)

        # Calculate weights (Kasa's weighting scheme)
        weights = distances / np.sum(distances)

        # Calculate updated center (weighted centroid)
        xc_new = np.sum(weights * points[:, 0])
        yc_new = np.sum(weights * points[:, 1])

        # Check for convergence
        if np.sqrt((xc_new - xc)**2 + (yc_new - yc)**2) < tolerance:
            break

        xc, yc = xc_new, yc_new

    # 3. Calculate radius
    radius = np.mean(distances)  

    return (xc, yc), radius


def detect_ellipse_fitzgibbon(points):
    """Detects ellipses using Fitzgibbon's algorithm (OpenCV)."""
    points = points.astype(np.float32)
    if len(points) >= 5: 
        ellipse = cv2.fitEllipse(points)
        return ellipse
    else:
        return None

def detect_rectangle_hough_harris(points):
    """Detects rectangles using Hough Transform and Harris Corner Detector."""
    # Harris Corner Detection
    coords = corner_harris(points, method='k', k=0.04)
    corner_indices = np.argwhere(coords > 0.01 * coords.max())
    corner_points = points[corner_indices.flatten()]

    if len(corner_points) < 4: 
        return None

    # Hough Line Transform
    h, theta, d = hough_line(corner_points)

    # Calculate threshold separately
    threshold = 0.5 * np.max(h) 

    accum, angles, dists = hough_line_peaks(h, theta, d, min_distance=5, min_angle=10, threshold=threshold)

    # Group lines to form rectangles 
    lines = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        if abs(np.sin(angle)) > 1e-6:  # Check if not a vertical line
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - 250 * np.cos(angle)) / np.sin(angle)  
            lines.append(((0, y0), (250, y1)))
        else: 
            x0 = dist / np.cos(angle) # Calculate x-coordinate for vertical lines
            lines.append(((x0, 0), (x0, 250))) # Append vertical line


    # Group lines based on similar angles (horizontal and vertical)
    horizontal_lines = []
    vertical_lines = []
    angle_threshold = np.deg2rad(5)  

    for p1, p2 in lines:
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        if abs(angle) < angle_threshold or abs(angle - np.pi) < angle_threshold:
            horizontal_lines.append((p1, p2))
        elif abs(angle - np.pi/2) < angle_threshold or abs(angle + np.pi/2) < angle_threshold:
            vertical_lines = []
    for h1, h2 in horizontal_lines:
        for v1, v2 in vertical_lines:
            # Calculate intersection point using line equations
            a1 = (h2[1] - h1[1]) / (h2[0] - h1[0])  # Slope of horizontal line
            b1 = h1[1] - a1 * h1[0]  # Y-intercept of horizontal line
            a2 = (v2[1] - v1[1]) / (v2[0] - v1[0])  # Slope of vertical line
            b2 = v1[1] - a2 * v1[0]  # Y-intercept of vertical line

            if abs(a1 - a2) < 1e-6:  # Lines are parallel (or very close to it)
                continue  # Skip this pair

            # Intersection point
            x_intersect = (b2 - b1) / (a1 - a2)
            y_intersect = a1 * x_intersect + b1

            # Check if intersection is within line segments
            if min(h1[0], h2[0]) <= x_intersect <= max(h1[0], h2[0]) and \
               min(v1[0], v2[0]) <= x_intersect <= max(v1[0], v2[0]) and \
               min(h1[1], h2[1]) <= y_intersect <= max(h1[1], h2[1]) and \
               min(v1[1], v2[1]) <= y_intersect <= max(v1[1], v2[1]):
                horizontal_lines.append(((x_intersect, y_intersect), h1, h2, v1, v2))

    # Select the best rectangle (closest to a perfect rectangle)
    best_rectangle = None
    best_score = float('inf')

    for intersection, h1, h2, v1, v2 in horizontal_lines:
        # Calculate side lengths 
        side1 = np.linalg.norm(np.array(h1[0]) - np.array(h2[0]))
        side2 = np.linalg.norm(np.array(v1[0]) - np.array(v2[0]))

        # Calculate a score based on side length difference (lower is better)
        score = abs(side1 - side2)

        if score < best_score:
            best_score = score
            best_rectangle = intersection, h1, h2, v1, v2

    if best_rectangle:
        _, h1, _, v1, _ = best_rectangle
        p1 = (v1[0], h1[1])
        p2 = (v1[0], h1[1])  # Adjusted to match solution format
        p3 = (v1[0], h1[1])  # Adjusted to match solution format
        p4 = (v1[0], h1[1])  # Adjusted to match solution format
        return p1, p2, p3, p4
    else:
        return None

def detect_star_rst(points):
    """Detects star shapes using a simplified radial symmetry approach."""
    # Calculate centroid
    centroid = np.mean(points, axis=0)

    # Calculate distances from centroid to each point
    distances = np.linalg.norm(points - centroid, axis=1)

    # Calculate angles from centroid to each point
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort points by angle
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    sorted_distances = distances[sorted_indices]

    # Calculate the difference in distances between consecutive points
    distance_diffs = np.diff(sorted_distances)

    # Look for alternating peaks (improved logic)
    peak_threshold = 0.2 * np.std(distance_diffs)
    num_points = 0
    inner_radius = float('inf')
    outer_radius = 0

    for i in range(len(distance_diffs) - 1):
        if distance_diffs[i] > peak_threshold and distance_diffs[i + 1] < -peak_threshold:
            num_points += 1
            inner_radius = min(inner_radius, sorted_distances[i + 1])
            outer_radius = max(outer_radius, sorted_distances[i])

    if 4 <= num_points <= 12:
        return centroid, inner_radius, outer_radius, num_points
    else:
        return None

# Function to generate polylines from shape parameters
def shape_to_polyline(shape_type, *params):
    """Generates a polyline representation of a shape given its parameters."""
    points = []
    if shape_type == "Line":
        slope, intercept = params
        x = np.linspace(0, 250, 100)  # Adjust range as needed
        y = slope * x + intercept
        points = np.column_stack((x, y))
    elif shape_type == "Circle":
        cx, cy, radius = params
        theta = np.linspace(0, 2*np.pi, 100)
        x = radius * np.cos(theta) + cx
        y = radius * np.sin(theta) + cy
        points = np.column_stack((x, y))
    elif shape_type == "Ellipse":
        cx, cy, major_axis, minor_axis, angle = params
        theta = np.linspace(0, 2*np.pi, 100)
        x = major_axis/2 * np.cos(theta) * np.cos(np.deg2rad(angle)) - minor_axis/2 * np.sin(theta) * np.sin(np.deg2rad(angle)) + cx
        y = major_axis/2 * np.cos(theta) * np.sin(np.deg2rad(angle)) + minor_axis/2 * np.sin(theta) * np.cos(np.deg2rad(angle)) + cy
        points = np.column_stack((x, y))
    elif shape_type == "Rectangle":
        p1, p2, p3, p4 = params # Unpack the corner points
        points = np.array([p1, p2, p3, p4, p1]) # Add the first point again to close the rectangle
    elif shape_type == "Star":
        cx, cy, inner_radius, outer_radius, num_points = params
        theta = np.linspace(0, 2*np.pi, num_points * 2, endpoint=False)
        x = np.zeros(num_points * 2)
        y = np.zeros(num_points * 2)
        x[::2] = outer_radius * np.cos(theta[::2]) + cx
        y[::2] = outer_radius * np.sin(theta[::2]) + cy
        x[1::2] = inner_radius * np.cos(theta[1::2]) + cx
        y[1::2] = inner_radius * np.sin(theta[1::2]) + cy
        points = np.column_stack((x, y))
    return points

def regularize_shapes(shapes):
    """
    Regularizes shapes in the input list.

    Args:
        shapes (list): A list of polylines (each polyline represented as 
                      a NumPy array of points).

    Returns:
        list: A list of regularized shapes, each represented as a list:
              [Shape_Type, Shape_ID, Param1, Param2, ...]
    """
    regularized_shapes = []
    for shape_id, shape_polylines in enumerate(shapes):  # Iterate over list, using index as shape_id
        # Assume each shape has only one polyline (modify if needed)
        points = shape_polylines[0]

        line = detect_line(points)
        circle = detect_circle_kasa(points) 
        ellipse = detect_ellipse_fitzgibbon(points)
        rectangle = detect_rectangle_hough_harris(points)
        star = detect_star_rst(points)

        if line:
            regularized_shapes.append(["Line", shape_id, line[0], line[1]])
        elif circle:
            regularized_shapes.append(["Circle", shape_id, circle[0][0], circle[0][1], circle[1]])
        elif ellipse:
            ((cx, cy), (major_axis, minor_axis), angle) = ellipse
            regularized_shapes.append(["Ellipse", shape_id, cx, cy, major_axis, minor_axis, angle])
        elif rectangle:
            # ... (Add logic to output rectangle parameters) ...
            p1, p2, p3, p4 = rectangle
            regularized_shapes.append(["Rectangle", shape_id, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]])
        elif star:
            # ... (Add logic to output star parameters) ...
            cx, cy, inner_radius, outer_radius, num_points = star
            regularized_shapes.append(["Star", shape_id, cx, cy, inner_radius, outer_radius, num_points])
        else:
            regularized_shapes.append(["Irregular", shape_id])

    return regularized_shapes

# Function to calculate accuracy
def calculate_accuracy(output_csv, solution_csv):
    """Calculates the accuracy of shape detection by comparing CSV files."""
    df_output = pd.read_csv(output_csv)
    df_solution = pd.read_csv(solution_csv)

    # Merge DataFrames based on shape ID
    df_merged = pd.merge(df_output, df_solution, on="Shape_ID", suffixes=("_output", "_solution"))

    # Calculate accuracy for each shape type
    accuracy = {}
    for shape_type in df_merged["Shape_output"].unique():
        df_shape = df_merged[df_merged["Shape_output"] == shape_type]
        correct = np.sum(np.allclose(df_shape[["Param1_output", "Param2_output", "Param3_output", "Param4_output",
                                               "Param5_output", "Param6_output", "Param7_output", "Param8_output",
                                               "Param9_output", "Param10_output"]],
                                     df_shape[["Param1_solution", "Param2_solution", "Param3_solution", "Param4_solution",
                                               "Param5_solution", "Param6_solution", "Param7_solution", "Param8_solution",
                                               "Param9_solution", "Param10_solution"]], atol=1e-6))
        total = len(df_shape)
        accuracy[shape_type] = correct / total if total > 0 else 0

    overall_accuracy = np.mean(list(accuracy.values()))
    return overall_accuracy, accuracy

# Main Execution

if __name__ == "__main__":
    csv_path = "Test_cases/isolated.csv" 
    output_csv = "regularized_shapes.csv"
    solution_csv = "Test_cases/isolated_sol.csv"

    shapes = read_csv(csv_path)
    plot(shapes)  # Visualize original shapes

    regularized_shapes = regularize_shapes(shapes)

    # Save regularized shapes to a CSV (Choose Method 1 or Method 2)

    # *** Method 1: Fill missing parameters with None ***
    #df_regularized = pd.DataFrame(regularized_shapes, columns=["Shape", "Shape_ID", "Param1", "Param2", "Param3", "Param4", "Param5", "Param6", "Param7", "Param8", "Param9", "Param10"]) 
    #df_regularized.to_csv("regularized_shapes.csv", index=False)

    # *** Method 2: Use a flexible DataFrame structure ***
    df_regularized = pd.DataFrame(columns=["Shape", "Shape_ID"]) 
    for shape in regularized_shapes:
        shape_type = shape[0]
        shape_id = shape[1]
        params = shape[2:]

        new_row = {"Shape": shape_type, "Shape_ID": shape_id}
        for i, param in enumerate(params):
            new_row[f"Param{i+1}"] = param

        # Use pd.concat() to add the new row
        df_regularized = pd.concat([df_regularized, pd.DataFrame([new_row])], ignore_index=True)

    df_regularized.to_csv("regularized_shapes.csv", index=False)
    
    # Visualization of Regularized Shapes
    regularized_polylines = []
    for shape in regularized_shapes:
        shape_type = shape[0]
        params = shape[2:]  # Exclude shape type and ID
        polyline = shape_to_polyline(shape_type, *params)
        regularized_polylines.append([polyline])  # Encapsulate in a list for plot()

    # Choose your visualization method:
    plot(regularized_polylines)  # Using your plot() function
    # OR
    # polylines2svg(regularized_polylines, "regularized_shapes.svg")  # Using your svg function

    # Calculate Accuracy
    overall_accuracy, shape_accuracy = calculate_accuracy(output_csv, solution_csv)

    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    for shape_type, acc in shape_accuracy.items():
        print(f"{shape_type} Accuracy: {acc:.2%}")