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
        # Check if it's a star
        if is_star(approx):
            return "Star"
        
        # Fit an ellipse to the contour
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        
        # Classify based on eccentricity
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
    # Get the exact dimensions from the original shape
    x_min, y_min = np.min(XY, axis=0)
    x_max, y_max = np.max(XY, axis=0)
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
    width = x_max - x_min
    height = y_max - y_min
    
    # Generate points for the ellipse using the exact dimensions
    t = np.linspace(0, 2 * np.pi, 200)
    x = center[0] + width/2 * np.cos(t)
    y = center[1] + height/2 * np.sin(t)
    
    return np.column_stack((x, y))

def regularize_rectangle(XY):
    # Get the exact dimensions from the original shape
    x_min, y_min = np.min(XY, axis=0)
    x_max, y_max = np.max(XY, axis=0)
    
    # Create regularized rectangle using original dimensions and position
    regularized = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min]  # Close the rectangle
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
    
    # Calculate radii
    radii = np.linalg.norm(XY - centroid, axis=1)
    outer_radius = np.max(radii)
    inner_radius = np.min(radii)
    
    # Generate points for the star
    angles = np.linspace(0, 2*np.pi, 2*num_points, endpoint=False)
    
    star_points = []
    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = centroid[0] + r * np.cos(angle)
        y = centroid[1] + r * np.sin(angle)
        star_points.append([x, y])
    
    star_points.append(star_points[0])  # Close the shape
    
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
    
    return shape, regularized_XY

def draw_symmetry_lines(ax, shape, reg_XY, color):
    def plot_line(ax, start, end, color):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linestyle='--', linewidth=1)

    center = np.mean(reg_XY, axis=0)
    
    if shape == "Ellipse":
        # Find the bounding box
        x_min, y_min = np.min(reg_XY, axis=0)
        x_max, y_max = np.max(reg_XY, axis=0)
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        
        # Draw major axis
        ax.plot([x_min, x_max], [center_y, center_y], color=color, linestyle='--', linewidth=1)
        
        # Draw minor axis
        ax.plot([center_x, center_x], [y_min, y_max], color=color, linestyle='--', linewidth=1)
        
        print("Ellipse has 2 lines of symmetry.")

    elif shape == "Rectangle":
        x_min, y_min = np.min(reg_XY, axis=0)
        x_max, y_max = np.max(reg_XY, axis=0)
        
        # Vertical symmetry line
        plot_line(ax, ((x_min + x_max)/2, y_min), ((x_min + x_max)/2, y_max), color)
        
        # Horizontal symmetry line
        plot_line(ax, (x_min, (y_min + y_max)/2), (x_max, (y_min + y_max)/2), color)
        
        print("Rectangle has 2 lines of symmetry.")

    elif shape == "Star":
        center = np.mean(reg_XY, axis=0)
        num_points = len(reg_XY) // 2  # Assuming the star has an even number of points
        
        for i in range(num_points):
            # Connect opposite points through the center
            start_point = reg_XY[i]
            end_point = reg_XY[(i + num_points) % len(reg_XY)]
            
            ax.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    color=color, linestyle='--', linewidth=1)
        
        print(f"Star has {num_points} lines of symmetry.")

    else:
        print(f"No specific symmetry lines defined for {shape}.")


def plot_shapes_with_labels(path_XYs, regularized_XYs):
    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(16, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for a in ax:
        a.set_aspect('equal')

    for i, (orig_XYs, reg_XYs) in enumerate(zip(path_XYs, regularized_XYs)):
        c = colours[i % len(colours)]
        for j, (XY, reg_XY) in enumerate(zip(orig_XYs, reg_XYs)):
            shape, _ = detect_and_regularize_shape(XY)
            
            ax[0].plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
            cx, cy = np.mean(XY, axis=0)
            ax[0].text(cx, cy, f"Original {shape} {j+1}", fontsize=12, ha='center', va='center', color='black')
            
            ax[1].plot(reg_XY[:, 0], reg_XY[:, 1], c=c, linewidth=2)
            cx, cy = np.mean(reg_XY, axis=0)
            ax[1].text(cx, cy, f"Regularized {shape} {j+1}", fontsize=12, ha='center', va='center', color='black')
            
            draw_symmetry_lines(ax[1], shape, reg_XY, color='black')

    ax[0].set_title('Original Shapes')
    ax[1].set_title('Regularized Shapes')

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
