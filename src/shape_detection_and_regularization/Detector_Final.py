import numpy as np
import matplotlib.pyplot as plt # type: ignore
import cv2
import svgwrite # type: ignore
import cairosvg # type: ignore
import csv

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

def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ab = b - a
    cb = b - c
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def detect_shape(XY):
    contour = np.array(XY, dtype=np.int32).reshape((-1, 1, 2))
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if cv2.contourArea(contour) < 100:
        return " "

    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
        return shape

    elif len(approx) > 5:
        if is_star(approx):
            return "Star"
        
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        
        if eccentricity < 0.1:
            return "Circle"
        elif 0.1 <= eccentricity < 0.8:
            return "Ellipse"
        else:
            return " "
    else:
        return " "

def is_star(approx):
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

def kasa_circle_fitting(XY):
    x = XY[:, 0]
    y = XY[:, 1]
    A = np.array([x, y, np.ones_like(x)]).T
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

def regularize_circle(XY):
    xc, yc, r = kasa_circle_fitting(XY)
    t = np.linspace(0, 2 * np.pi, 100)
    x_fit = xc + r * np.cos(t)
    y_fit = yc + r * np.sin(t)
    return np.vstack((x_fit, y_fit)).T

def fit_ellipse_direct_method(XY):
    X = XY[:, 0]
    Y = XY[:, 1]
    D = np.vstack((X**2, X*Y, Y**2, X, Y, np.ones_like(X))).T

    S = np.dot(D.T, D)
    _, V = np.linalg.eig(S)

    min_index = np.argmin(_)
    ellipse_params = V[:, min_index]

    a, b, c, d, e, f = ellipse_params
    return a, b, c, d, e, f

def get_ellipse_params_from_general_form(a, b, c, d, e, f):
    M = np.array([[a, b/2], [b/2, c]])
    eigenvalues, eigenvectors = np.linalg.eig(M)
    major_axis = np.sqrt(2 / np.min(eigenvalues))
    minor_axis = np.sqrt(2 / np.max(eigenvalues))
    center_x = (d * e - 2 * f * c) / (4 * a * c - b**2)
    center_y = (d * b - 2 * a * e) / (4 * a * c - b**2)

    rotation_angle = 0.5 * np.arctan2(b, a - c)

    return center_x, center_y, major_axis, minor_axis, rotation_angle

def regularize_ellipse(XY):
    a, b, c, d, e, f = fit_ellipse_direct_method(XY)
    center_x, center_y, major_axis, minor_axis, rotation_angle = get_ellipse_params_from_general_form(a, b, c, d, e, f)

    bounding_box = cv2.boundingRect(XY.astype(np.int32))
    max_width = bounding_box[2]
    max_height = bounding_box[3]

    major_axis = min(major_axis, max_width / 2)
    minor_axis = min(minor_axis, max_height / 2)

    t = np.linspace(0, 2 * np.pi, 100)
    x_fit = center_x + major_axis * np.cos(t) * np.cos(rotation_angle) - minor_axis * np.sin(t) * np.sin(rotation_angle)
    y_fit = center_y + major_axis * np.cos(t) * np.sin(rotation_angle) + minor_axis * np.sin(t) * np.cos(rotation_angle)

    regularized_ellipse = np.vstack((x_fit, y_fit)).T

    reg_bbox = cv2.boundingRect(regularized_ellipse.astype(np.int32))
    dx = bounding_box[0] - reg_bbox[0]
    dy = bounding_box[1] - reg_bbox[1]
    scale_x = bounding_box[2] / reg_bbox[2]
    scale_y = bounding_box[3] / reg_bbox[3]

    regularized_ellipse[:, 0] = regularized_ellipse[:, 0] * scale_x + dx
    regularized_ellipse[:, 1] = regularized_ellipse[:, 1] * scale_y + dy

    return regularized_ellipse

def regularize_rectangle(XY):
    XY = np.array(XY, dtype=np.float32).reshape((-1, 1, 2))
    
    rect = cv2.minAreaRect(XY)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    width = np.linalg.norm(box[1] - box[0])
    height = np.linalg.norm(box[3] - box[0])
    
    center = np.mean(box, axis=0)
    
    half_width, half_height = width/2, height/2
    regularized = np.array([
        [center[0] - half_width, center[1] - half_height],
        [center[0] + half_width, center[1] - half_height],
        [center[0] + half_width, center[1] + half_height],
        [center[0] - half_width, center[1] + half_height],
        [center[0] - half_width, center[1] - half_height]  
    ])
    
    return regularized

def regularize_square(XY):
    X = XY[:, 0]
    Y = XY[:, 1]
    A = np.vstack([X, np.ones(len(X))]).T
    m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
    reg_XY = np.column_stack((X, m * X + c))
    width = np.max(X) - np.min(X)
    height = np.max(Y) - np.min(Y)
    side_length = min(width, height)

    x_min = np.min(X)
    y_min = np.min(Y)
    reg_XY = np.array([
        [x_min, y_min],
        [x_min + side_length, y_min],
        [x_min + side_length, y_min + side_length],
        [x_min, y_min + side_length]
    ])

    return reg_XY

def regularize_star(XY, num_points=5):
    centroid = np.mean(XY, axis=0)
    
    radii = np.linalg.norm(XY - centroid, axis=1)
    avg_radius = np.mean(radii)
    
    angles = np.linspace(0, 2*np.pi, 2*num_points, endpoint=False)
    outer_radius = avg_radius
    inner_radius = avg_radius * 0.5
    
    star_points = []
    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = centroid[0] + r * np.cos(angle)
        y = centroid[1] + r * np.sin(angle)
        star_points.append([x, y])
    
    star_points.append(star_points[0])
    
    return np.array(star_points)

def detect_and_regularize_shape(XY):
    shape = detect_shape(XY)
    if shape == "Circle":
        regularized_XY = regularize_circle(XY)
    elif shape == "Ellipse":
        regularized_XY = regularize_ellipse(XY)
    elif shape == "Rectangle":
        regularized_XY = regularize_rectangle(XY)
    elif shape == "Square":
        regularized_XY = regularize_square(XY)
    elif shape == "Star":
        regularized_XY = regularize_star(XY)
    else:
        regularized_XY = XY
    
    original_area = cv2.contourArea(XY.astype(np.float32))
    regularized_area = cv2.contourArea(regularized_XY.astype(np.float32))
    scale_factor = np.sqrt(original_area / regularized_area)
    
    center = np.mean(regularized_XY, axis=0)
    regularized_XY = (regularized_XY - center) * scale_factor + center
    
    original_center = np.mean(XY, axis=0)
    regularized_center = np.mean(regularized_XY, axis=0)
    translation = original_center - regularized_center
    regularized_XY += translation
    
    return shape, regularized_XY

def draw_symmetry_lines(ax, shape, reg_XY, color):
    def clip_line_to_shape(line, shape_points):
        intersections = []
        for i in range(len(shape_points) - 1):
            x1, y1 = shape_points[i]
            x2, y2 = shape_points[i + 1]
            denominator = (y2 - y1) * (line[1][0] - line[0][0]) - (x2 - x1) * (line[1][1] - line[0][1])
            if denominator != 0:
                ua = ((x2 - x1) * (line[0][1] - y1) - (y2 - y1) * (line[0][0] - x1)) / denominator
                ub = ((line[1][0] - line[0][0]) * (line[0][1] - y1) - (line[1][1] - line[0][1]) * (line[0][0] - x1)) / denominator
                if 0 <= ua <= 1 and 0 <= ub <= 1:
                    intersection_x = line[0][0] + ua * (line[1][0] - line[0][0])
                    intersection_y = line[0][1] + ua * (line[1][1] - line[0][1])
                    intersections.append([intersection_x, intersection_y])
        return intersections

    def plot_symmetry_line(ax, line, shape_points, color):
        intersections = clip_line_to_shape(line, shape_points)
        if len(intersections) == 2:
            ax.plot([intersections[0][0], intersections[1][0]], [intersections[0][1], intersections[1][1]], color=color, linestyle='dotted')

    symmetry_lines_count = 0

    if shape in ["Circle", "Ellipse"]:
        symmetry_lines_count = 2  
        center = np.mean(reg_XY, axis=0)
        major_axis_angle = np.arctan2(reg_XY[1, 1] - reg_XY[0, 1], reg_XY[1, 0] - reg_XY[0, 0])
        minor_axis_angle = major_axis_angle + np.pi / 2
        
        major_axis_line = [[center[0] - 1000 * np.cos(major_axis_angle), center[1] - 1000 * np.sin(major_axis_angle)], 
                           [center[0] + 1000 * np.cos(major_axis_angle), center[1] + 1000 * np.sin(major_axis_angle)]]
        minor_axis_line = [[center[0] - 1000 * np.cos(minor_axis_angle), center[1] - 1000 * np.sin(minor_axis_angle)], 
                           [center[0] + 1000 * np.cos(minor_axis_angle), center[1] + 1000 * np.sin(minor_axis_angle)]]
        
        plot_symmetry_line(ax, major_axis_line, reg_XY, color)
        plot_symmetry_line(ax, minor_axis_line, reg_XY, color)

    elif shape in ["Rectangle", "Square"]:
        center = np.mean(reg_XY, axis=0)
        vertical_line = [[center[0], center[1] - 1000], [center[0], center[1] + 1000]]
        horizontal_line = [[center[0] - 1000, center[1]], [center[0] + 1000, center[1]]]
        
        plot_symmetry_line(ax, vertical_line, reg_XY, color)
        plot_symmetry_line(ax, horizontal_line, reg_XY, color)

        symmetry_lines_count = 2  

        if shape == "Square":
            diagonal_slope = np.tan(np.pi / 4)
            diagonal1 = [[center[0] - 1000 * diagonal_slope, center[1] - 1000], [center[0] + 1000 * diagonal_slope, center[1] + 1000]]
            diagonal2 = [[center[0] + 1000 * diagonal_slope, center[1] - 1000], [center[0] - 1000 * diagonal_slope, center[1] + 1000]]
            
            plot_symmetry_line(ax, diagonal1, reg_XY, color)
            plot_symmetry_line(ax, diagonal2, reg_XY, color)
            
            symmetry_lines_count = 4  

    elif shape == "Star":
        num_points = len(reg_XY) // 2
        symmetry_lines_count = num_points 
        center = np.mean(reg_XY, axis=0)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        for angle in angles:
            line = [[center[0] - 1000 * np.cos(angle), center[1] - 1000 * np.sin(angle)], 
                    [center[0] + 1000 * np.cos(angle), center[1] + 1000 * np.sin(angle)]]
            plot_symmetry_line(ax, line, reg_XY, color)

    print(f"{shape} has {symmetry_lines_count} lines of symmetry.")

def plot_shapes_with_labels(path_XYs, regularized_XYs):
    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(16, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, (orig_XYs, reg_XYs) in enumerate(zip(path_XYs, regularized_XYs)):
        c = colours[i % len(colours)]
        for j, (XY, reg_XY) in enumerate(zip(orig_XYs, reg_XYs)):
            shape, reg_XY = detect_and_regularize_shape(XY)
            
            ax[0].plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
            cx, cy = np.mean(XY, axis=0)
            ax[0].text(cx, cy, f"Original {shape} {j+1}", fontsize=12, ha='center', color='black')
            
            ax[1].plot(reg_XY[:, 0], reg_XY[:, 1], c=c, linewidth=2)
            cx, cy = np.mean(reg_XY, axis=0)
            ax[1].text(cx, cy, f"Regularized {shape} {j+1}", fontsize=12, ha='center', color='black')

            draw_symmetry_lines(ax[1], shape, reg_XY, color='black')

    ax[0].set_title('Original Shapes')
    ax[1].set_title('Regularized Shapes')
    for a in ax:
        a.set_aspect('equal')

    plt.show()

def polylines2csv(paths_XYs, csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for path_XYs in paths_XYs:
            for i, XY in enumerate(path_XYs):
                for point in XY:
                    writer.writerow([i] + point.tolist())

def main():
    input_csv = "Test_cases/isolated.csv"
    output_csv = "Test_cases/regularized_shapes.csv"
    path_XYs = read_csv(input_csv)
    regularized_XYs = []

    for path in path_XYs:
        reg_XYs = []
        for XY in path:
            _, reg_XY = detect_and_regularize_shape(XY)
            reg_XYs.append(reg_XY)
        regularized_XYs.append(reg_XYs)

    plot_shapes_with_labels(path_XYs, regularized_XYs)
    polylines2csv(regularized_XYs, output_csv)

if __name__ == "__main__":
    main()
