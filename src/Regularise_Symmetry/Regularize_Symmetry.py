import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv

def read_csv(csv_path):
    """
    Reads a CSV file and organizes the points into paths.

    Parameters:
    - csv_path (str): The file path of the CSV file.

    Returns:
    - path_XYs (list of list of numpy arrays): A list of paths where each path is a list of arrays representing points.
    """
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

def detect_shape(XY):
    """
    Detects the shape type from the provided points.

    Parameters:
    - XY (numpy array): An array of points representing a shape.

    Returns:
    - shape (str): The detected shape type, such as "Circle", "Ellipse", "Rectangle", "Square", "Star", or "Straight Line".
    """
    contour = np.array(XY, dtype=np.int32).reshape((-1, 1, 2))
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    area = cv2.contourArea(contour)
    
    if area < 100:
        return " "

    if is_straight_line(XY):
        return "Straight Line"

    if len(approx) == 4:
        if is_square(contour):
            return "Square"
        else:
            return "Rectangle"

    elif len(approx) > 5:
        if is_star(approx):
            return "Star"
        
        circularity = 4 * np.pi * area / (peri * peri)
        
        if circularity > 0.95:
            return "Circle"
        elif 0.8 <= circularity < 0.95:
            ellipse = cv2.fitEllipse(contour)
            major_axis, minor_axis = max(ellipse[1]), min(ellipse[1])
            if major_axis / minor_axis > 1.05:
                return "Ellipse"
            else:
                return "Circle"
    
    return " "

def is_square(contour, tolerance=0.05):
    """
    Checks if the contour forms a square.

    Parameters:
    - contour (numpy array): The contour of a shape.
    - tolerance (float, optional): Tolerance for the square check.

    Returns:
    - is_square_shape (bool): Whether the contour is a square or not.
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    width = np.linalg.norm(box[0] - box[1])
    height = np.linalg.norm(box[1] - box[2])
    aspect_ratio = min(width, height) / max(width, height)

    contour_area = cv2.contourArea(contour)
    rect_area = width * height
    area_ratio = contour_area / rect_area

    is_square_shape = abs(1 - aspect_ratio) < tolerance
    is_filled = area_ratio > (1 - tolerance)

    return is_square_shape and is_filled

def is_star(approx):
    """
    Checks if the contour forms a star shape.

    Parameters:
    - approx (numpy array): The approximated contour points.

    Returns:
    - is_star_shape (bool): Whether the contour is a star or not.
    """
    num_vertices = len(approx)
    if num_vertices > 8:
        angles = []
        for i in range(num_vertices):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % num_vertices][0]
            p3 = approx[(i + 2) % num_vertices][0]
            angle = calculate_angle(p1, p2, p3)
            angles.append(angle)
        
        sharp_angles = [angle for angle in angles if angle < 60 or angle > 300]
        if len(sharp_angles) >= 5:
            return True

    return False

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle between three points.

    Parameters:
    - p1, p2, p3 (numpy arrays): Three points to form an angle.

    Returns:
    - angle (float): The angle in degrees.
    """
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ab = b - a
    cb = b - c
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def is_straight_line(XY, tolerance=0.05):
    """
    Checks if the points form a straight line.

    Parameters:
    - XY (numpy array): An array of points representing a line.
    - tolerance (float, optional): Tolerance for the straight line check.

    Returns:
    - is_straight_line (bool): Whether the points form a straight line or not.
    """
    if len(XY) < 3:
        return True
    
    total_length = np.sum(np.sqrt(np.sum(np.diff(XY, axis=0)**2, axis=1)))
    start_to_end = np.linalg.norm(XY[-1] - XY[0])
    return (total_length - start_to_end) / total_length < tolerance

def regularize_circle(XY):
    """
    Regularizes the points to form a perfect circle.

    Parameters:
    - XY (numpy array): An array of points representing a circle.

    Returns:
    - reg_XY (numpy array): Regularized circle points.
    """
    xc, yc, r = kasa_circle_fitting(XY)
    t = np.linspace(0, 2 * np.pi, 100)
    x_fit = xc + r * np.cos(t)
    y_fit = yc + r * np.sin(t)
    return np.vstack((x_fit, y_fit)).T

def kasa_circle_fitting(XY):
    """
    Fits a circle to the points using Kasa's method.

    Parameters:
    - XY (numpy array): An array of points representing a circle.

    Returns:
    - xc, yc, r (floats): Circle center coordinates and radius.
    """
    x = XY[:, 0]
    y = XY[:, 1]
    A = np.array([x, y, np.ones_like(x)]).T
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

def regularize_ellipse(XY):
    """
    Regularizes the points to form a perfect ellipse.

    Parameters:
    - XY (numpy array): An array of points representing an ellipse.

    Returns:
    - reg_XY (numpy array): Regularized ellipse points.
    """
    x_min, y_min = np.min(XY, axis=0)
    x_max, y_max = np.max(XY, axis=0)
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
    width = x_max - x_min
    height = y_max - y_min
    t = np.linspace(0, 2 * np.pi, 200)
    x = center[0] + width/2 * np.cos(t)
    y = center[1] + height/2 * np.sin(t)
    
    return np.column_stack((x, y))

def regularize_rectangle(XY):
    """
    Regularizes the points to form a perfect rectangle.

    Parameters:
    - XY (numpy array): An array of points representing a rectangle.

    Returns:
    - reg_XY (numpy array): Regularized rectangle points.
    """
    x_min, y_min = np.min(XY, axis=0)
    x_max, y_max = np.max(XY, axis=0)

    regularized = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min]  
    ])
    
    return regularized

def regularize_square(XY):
    """
    Regularizes the points to form a perfect square.

    Parameters:
    - XY (numpy array): An array of points representing a square.

    Returns:
    - reg_XY (numpy array): Regularized square points.
    """
    x_min, y_min = np.min(XY, axis=0)
    x_max, y_max = np.max(XY, axis=0)

    side_length = min(x_max - x_min, y_max - y_min)

    reg_XY = np.array([
        [x_min, y_min],
        [x_min + side_length, y_min],
        [x_min + side_length, y_min + side_length],
        [x_min, y_min + side_length],
        [x_min, y_min] 
    ])

    return reg_XY

def regularize_star(XY, num_points=5):
    """
    Regularizes the points to form a perfect star.

    Parameters:
    - XY (numpy array): An array of points representing a star.
    - num_points (int, optional): Number of points in the star.

    Returns:
    - reg_XY (numpy array): Regularized star points.
    """
    t = np.linspace(0, 2 * np.pi, 2 * num_points, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    points = np.column_stack((x, y))
    
    # Normalize to ensure symmetry
    reg_XY = points / np.max(np.sqrt(x**2 + y**2))

    return reg_XY

def regularize_straight_line(XY):
    """
    Regularizes the points to form a perfect straight line.

    Parameters:
    - XY (numpy array): An array of points representing a line.

    Returns:
    - reg_XY (numpy array): Regularized straight line points.
    """
    x_min, y_min = np.min(XY, axis=0)
    x_max, y_max = np.max(XY, axis=0)

    reg_XY = np.array([
        [x_min, y_min],
        [x_max, y_max]
    ])

    return reg_XY

def detect_and_regularize_shape(XY):
    """
    Detects the shape and returns the regularized points.

    Parameters:
    - XY (numpy array): An array of points representing a shape.

    Returns:
    - shape (str): The detected shape type.
    - reg_XY (numpy array): Regularized shape points.
    """
    shape = detect_shape(XY)
    
    if shape == "Circle":
        reg_XY = regularize_circle(XY)
    elif shape == "Ellipse":
        reg_XY = regularize_ellipse(XY)
    elif shape == "Rectangle":
        reg_XY = regularize_rectangle(XY)
    elif shape == "Square":
        reg_XY = regularize_square(XY)
    elif shape == "Star":
        reg_XY = regularize_star(XY)
    elif shape == "Straight Line":
        reg_XY = regularize_straight_line(XY)
    else:
        reg_XY = XY
    
    return shape, reg_XY

def calculate_symmetry_lines(XY, shape_type):
    """
    Calculates symmetry lines for the given shape.

    Parameters:
    - XY (numpy array): An array of points representing a shape.
    - shape_type (str): The shape type ("Circle", "Ellipse", "Square", "Rectangle", "Star").

    Returns:
    - lines (list of tuples): List of symmetry lines defined by their start and end points.
    """
    lines = []

    if shape_type in ["Square", "Rectangle"]:
        x_min, y_min = np.min(XY, axis=0)
        x_max, y_max = np.max(XY, axis=0)
        center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
        
        if shape_type == "Square":
            lines = [
                [(x_min, center[1]), (x_max, center[1])],
                [(center[0], y_min), (center[0], y_max)],
                [(x_min, y_min), (x_max, y_max)],
                [(x_min, y_max), (x_max, y_min)]
            ]
        elif shape_type == "Rectangle":
            lines = [
                [(x_min, center[1]), (x_max, center[1])],
                [(center[0], y_min), (center[0], y_max)]
            ]

    elif shape_type == "Circle":
        xc, yc, r = kasa_circle_fitting(XY)
        for angle in np.linspace(0, 2 * np.pi, 8):
            x_start = xc + r * np.cos(angle)
            y_start = yc + r * np.sin(angle)
            x_end = xc + r * np.cos(angle + np.pi)
            y_end = yc + r * np.sin(angle + np.pi)
            lines.append([(x_start, y_start), (x_end, y_end)])

    elif shape_type == "Ellipse":
        ellipse = cv2.fitEllipse(np.array(XY, dtype=np.int32).reshape((-1, 1, 2)))
        center, axes, angle = ellipse
        a, b = axes
        for angle in np.linspace(0, 2 * np.pi, 8):
            x_start = center[0] + a * np.cos(angle) * np.cos(np.radians(angle)) - b * np.sin(angle) * np.sin(np.radians(angle))
            y_start = center[1] + a * np.cos(angle) * np.sin(np.radians(angle)) + b * np.sin(angle) * np.cos(np.radians(angle))
            x_end = center[0] + a * np.cos(angle + np.pi) * np.cos(np.radians(angle)) - b * np.sin(angle + np.pi) * np.sin(np.radians(angle))
            y_end = center[1] + a * np.cos(angle + np.pi) * np.sin(np.radians(angle)) + b * np.sin(angle + np.pi) * np.cos(np.radians(angle))
            lines.append([(x_start, y_start), (x_end, y_end)])

    elif shape_type == "Star":
        center = np.mean(XY, axis=0)
        for i in range(len(XY)):
            x_start = XY[i, 0]
            y_start = XY[i, 1]
            x_end = center[0] + (x_start - center[0]) * 2
            y_end = center[1] + (y_start - center[1]) * 2
            lines.append([(x_start, y_start), (x_end, y_end)])

    return lines

def plot_shapes(path_XYs, regularized_XYs):
    """
    Plots original and regularized shapes with symmetry lines.

    Parameters:
    - path_XYs (list of list of numpy arrays): Original paths of points.
    - regularized_XYs (list of list of numpy arrays): Regularized paths of points.
    """
    fig, axs = plt.subplots(len(path_XYs), 2, figsize=(12, 6 * len(path_XYs)))
    
    for i, (paths, reg_paths) in enumerate(zip(path_XYs, regularized_XYs)):
        for path in paths:
            axs[i, 0].plot(path[:, 0], path[:, 1], 'o-')
        axs[i, 0].set_title(f"Original Shape {i+1}")
        axs[i, 0].set_aspect('equal')
        
        for reg_path in reg_paths:
            shape, reg_XY = detect_and_regularize_shape(reg_path)
            axs[i, 1].plot(reg_XY[:, 0], reg_XY[:, 1], 'o-')
            lines = calculate_symmetry_lines(reg_XY, shape)
            for line in lines:
                axs[i, 1].plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'r--')
            axs[i, 1].set_title(f"Regularized Shape {i+1}")
            axs[i, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def draw_symmetry_lines(ax, shape, reg_XY, color):
    """
    Draws symmetry lines on the given axis for the specified shape.

    Parameters:
    - ax (matplotlib axis): The axis to draw on.
    - shape (str): The shape type.
    - reg_XY (numpy array): Regularized shape points.
    - color (str): Color of the symmetry lines.
    """
    lines = calculate_symmetry_lines(reg_XY, shape)
    for line in lines:
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color=color)

def polylines2csv(paths_XYs, csv_path):
    """
    Writes the regularized shapes to a CSV file.

    Parameters:
    - paths_XYs (list of list of numpy arrays): Regularized paths of points.
    - csv_path (str): The file path to save the CSV file.
    """
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for path_XYs in paths_XYs:
            for path in path_XYs:
                for point in path:
                    csvwriter.writerow(point)
                csvwriter.writerow([])  

def main():
    """
    Main function to execute the script. Reads input CSV, detects and regularizes shapes,
    plots the results, and writes the regularized shapes to a new CSV file.
    """
    csv_input = 'shapes_input.csv'
    csv_output = 'shapes_regularized.csv'
    
    path_XYs = read_csv(csv_input)
    
    regularized_XYs = []
    for paths in path_XYs:
        reg_paths = []
        for path in paths:
            shape, reg_XY = detect_and_regularize_shape(path)
            reg_paths.append(reg_XY)
        regularized_XYs.append(reg_paths)
    
    plot_shapes(path_XYs, regularized_XYs)
    polylines2csv(regularized_XYs, csv_output)

if __name__ == "__main__":
    main()
