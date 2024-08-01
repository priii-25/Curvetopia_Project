import numpy as np
import scipy.special
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

def bezier_curve(points, t):
    n = len(points) - 1
    return sum(
        scipy.special.comb(n, i) * (1 - t) ** (n - i) * t ** i * points[i]
        for i in range(n + 1)
    )

def bezier_arc_completion(P_A, theta_A, P_B, theta_B, n_arcs=20, n_control_points=6):
    def objective_function(params):
        control_points = np.reshape(params, (-1, 2))
        t_values = np.linspace(0, 1, n_arcs)
        curve_points = np.array([bezier_curve(control_points, t) for t in t_values])
        
        # Endpoint errors
        error_A = np.linalg.norm(curve_points[0] - P_A)
        error_B = np.linalg.norm(curve_points[-1] - P_B)
        
        # Tangent errors
        tangent_A = curve_points[1] - curve_points[0]
        tangent_B = curve_points[-1] - curve_points[-2]
        angle_error_A = np.abs(np.arctan2(tangent_A[1], tangent_A[0]) - theta_A)
        angle_error_B = np.abs(np.arctan2(tangent_B[1], tangent_B[0]) - theta_B)
        
        # Curvature continuity
        curvature_A = np.linalg.norm(np.cross(tangent_A, curve_points[2] - curve_points[1]))
        curvature_B = np.linalg.norm(np.cross(tangent_B, curve_points[-3] - curve_points[-2]))
        curvature_error = np.abs(curvature_A - curvature_B)
        
        # Total curvature (for smoothness)
        total_curvature = sum(np.linalg.norm(np.cross(curve_points[i+1] - curve_points[i], 
                                                      curve_points[i+2] - curve_points[i+1])) 
                              for i in range(len(curve_points)-2))
        
        return error_A + error_B + angle_error_A + angle_error_B + 0.1 * curvature_error + 0.01 * total_curvature

    # Initialize control points along the line connecting P_A and P_B
    initial_control_points = np.linspace(P_A, P_B, n_control_points)
    
    # Adjust initial control points to match tangents
    initial_control_points[1] = P_A + np.array([np.cos(theta_A), np.sin(theta_A)])
    initial_control_points[-2] = P_B - np.array([np.cos(theta_B), np.sin(theta_B)])

    result = minimize(objective_function, initial_control_points.flatten(), method='L-BFGS-B')
    control_points = np.reshape(result.x, (-1, 2))
    t_values = np.linspace(0, 1, n_arcs)
    curve_points = np.array([bezier_curve(control_points, t) for t in t_values])
    return curve_points

def detect_occlusions(paths, threshold=10):
    occlusions = []
    for path in paths:
        for i in range(len(path) - 1):
            dist = np.linalg.norm(path[i+1][0] - path[i][-1])
            if dist > threshold:
                occlusions.append((path[i][-1], path[i+1][0]))
    return occlusions

def estimate_tangent(points, index):
    if index == 0:
        return points[1] - points[0]
    elif index == len(points) - 1:
        return points[-1] - points[-2]
    else:
        return points[index+1] - points[index-1]

def complete_occlusions(paths, occlusions):
    completed_paths = []
    for path in paths:
        completed_path = []
        for segment in path:
            completed_path.append(segment)
            for occlusion in occlusions:
                if np.allclose(segment[-1], occlusion[0]):
                    P_A = occlusion[0]
                    P_B = occlusion[1]
                    theta_A = np.arctan2(*(estimate_tangent(segment, -1)[::-1]))
                    theta_B = np.arctan2(*(estimate_tangent(path[path.index(segment)+1], 0)[::-1]))
                    completed_segment = bezier_arc_completion(P_A, theta_A, P_B, theta_B, n_arcs=20, n_control_points=6)
                    completed_path.append(completed_segment)
        completed_paths.append(completed_path)
    return completed_paths

def calculate_accuracy(completed_paths, solution_paths):
    total_error = 0
    total_points = 0
    for completed_path, solution_path in zip(completed_paths, solution_paths):
        for completed_segment, solution_segment in zip(completed_path, solution_path):
            if len(completed_segment) != len(solution_segment):
                min_length = min(len(completed_segment), len(solution_segment))
                completed_segment = completed_segment[:min_length]
                solution_segment = solution_segment[:min_length]
            error = np.linalg.norm(completed_segment - solution_segment, axis=1)
            total_error += np.sum(error)
            total_points += len(error)
    accuracy = total_error / total_points
    return accuracy

def plot_paths(paths, title=""):
    plt.figure(figsize=(10, 10))
    for path in paths:
        for segment in path:
            plt.plot(segment[:, 0], segment[:, 1])
    plt.title(title)
    plt.axis('equal')
    plt.show()

def plot_paths_comparison_three(paths1, paths2, paths3, title1="", title2="", title3=""):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    for path in paths1:
        for segment in path:
            plt.plot(segment[:, 0], segment[:, 1])
    plt.title(title1)
    plt.axis('equal')

    plt.subplot(1, 3, 2)
    for path in paths2:
        for segment in path:
            plt.plot(segment[:, 0], segment[:, 1])
    plt.title(title2)
    plt.axis('equal')

    plt.subplot(1, 3, 3)
    for path in paths3:
        for segment in path:
            plt.plot(segment[:, 0], segment[:, 1])
    plt.title(title3)
    plt.axis('equal')

    plt.show()

input_file = "occlusion1.csv"  
solution_file = "occlusion1_sol.csv"  
paths = read_csv(input_file)
solution_paths = read_csv(solution_file)

plot_paths(paths, "Original Paths")

occlusions = detect_occlusions(paths)
completed_paths = complete_occlusions(paths, occlusions)

accuracy = calculate_accuracy(completed_paths, solution_paths)
print(f"Accuracy: {accuracy}")

plot_paths_comparison_three(paths, completed_paths, solution_paths, title1="Original Paths", title2="Completed Paths", title3="Solution Paths")

