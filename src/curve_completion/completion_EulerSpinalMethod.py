import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

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

def euler_arc_spline_completion(P_A, theta_A, P_B, theta_B, n_arcs=10):
    def V(theta_j, Dtheta_j):
        return np.array([
            2 * np.sin(Dtheta_j / 2) / Dtheta_j * np.cos(theta_A + theta_j),
            2 * np.sin(Dtheta_j / 2) / Dtheta_j * np.sin(theta_A + theta_j)
        ])

    def compute_perturbations(s, k1):
        a = 2 * (theta_B - theta_A - n_arcs * k1 * s) / (n_arcs * (n_arcs - 1) * s**2)
        Dtheta = np.array([a * s**2 * (j - 1) + k1 * s for j in range(1, n_arcs + 1)])
        theta = np.cumsum(Dtheta) + theta_A
        phi = (theta[:-1] + theta[1:]) / 2 - theta_A
        V_x = np.array([V(phi[j], Dtheta[j])[0] for j in range(n_arcs - 1)])
        V_y = np.array([V(phi[j], Dtheta[j])[1] for j in range(n_arcs - 1)])
        A = np.sum(V_x**2) * np.sum(V_y**2) - np.sum(V_x * V_y)**2
        B = 2 * (P_B[0] - P_A[0] - s * np.sum(V_x))
        C = 2 * (P_B[1] - P_A[1] - s * np.sum(V_y))
        lambda1 = (np.sum(V_y**2) * B - np.sum(V_x * V_y) * C) / A
        lambda2 = (np.sum(V_x**2) * C - np.sum(V_x * V_y) * B) / A
        delta = -(lambda1 * V_x + lambda2 * V_y) / 2
        return delta, a, Dtheta

    def objective_function(params):
        s, k1 = params
        delta, _, _ = compute_perturbations(s, k1)
        return np.sum(delta**2)

    s_init = np.linalg.norm(np.array(P_B) - np.array(P_A)) / n_arcs
    k1_init = (theta_B - theta_A) / (n_arcs * s_init)
    bounds = [
        (np.linalg.norm(np.array(P_B) - np.array(P_A)) / n_arcs, None),
        (max(-2*np.pi/s_init, -2*np.pi + (theta_B - theta_A) / n_arcs),
         min(2*np.pi/s_init, 2*np.pi + (theta_B - theta_A) / n_arcs))
    ]
    
    result = minimize(objective_function, [s_init, k1_init], method='L-BFGS-B', bounds=bounds)
    s_opt, k1_opt = result.x
    delta_opt, _, Dtheta_opt = compute_perturbations(s_opt, k1_opt)
    
    curve_points = [P_A]
    theta = theta_A
    
    for i in range(n_arcs):
        s_i = s_opt + delta_opt[i % (n_arcs - 1)]
        theta += Dtheta_opt[i] / 2
        x = curve_points[-1][0] + s_i * np.cos(theta)
        y = curve_points[-1][1] + s_i * np.sin(theta)
        curve_points.append((x, y))
        theta += Dtheta_opt[i] / 2
    
    return np.array(curve_points)

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
                    completed_segment = euler_arc_spline_completion(P_A, theta_A, P_B, theta_B)
                    completed_path.append(completed_segment)
        completed_paths.append(completed_path)
    return completed_paths

def resample_path(path, num_points=100):
    """Resample a path to a fixed number of points using linear interpolation."""
    resampled_path = []
    for segment in path:
        if len(segment) > 1:
            distances = np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1))
            distances = np.insert(distances, 0, 0) / distances[-1]
            interpolator = interp1d(distances, segment, axis=0, kind='linear')
            resampled_path.append(interpolator(np.linspace(0, 1, num_points)))
        else:
            resampled_path.append(segment)
    return resampled_path

def plot_paths(paths, title=""):
    plt.figure(figsize=(10, 10))
    for path in paths:
        for segment in path:
            plt.plot(segment[:, 0], segment[:, 1])
    plt.title(title)
    plt.axis('equal')
    plt.show()

def plot_paths_comparison(paths1, paths2, title1="Paths 1", title2="Paths 2"):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    for path in paths1:
        for segment in path:
            ax[0].plot(segment[:, 0], segment[:, 1])
    ax[0].set_title(title1)
    ax[0].axis('equal')
    
    for path in paths2:
        for segment in path:
            ax[1].plot(segment[:, 0], segment[:, 1])
    ax[1].set_title(title2)
    ax[1].axis('equal')
    
    plt.show()

def plot_paths_comparison_three(paths1, paths2, paths3, title1="Paths 1", title2="Paths 2", title3="Paths 3"):
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    
    for path in paths1:
        for segment in path:
            ax[0].plot(segment[:, 0], segment[:, 1])
    ax[0].set_title(title1)
    ax[0].axis('equal')
    
    for path in paths2:
        for segment in path:
            ax[1].plot(segment[:, 0], segment[:, 1])
    ax[1].set_title(title2)
    ax[1].axis('equal')
    
    for path in paths3:
        for segment in path:
            ax[2].plot(segment[:, 0], segment[:, 1])
    ax[2].set_title(title3)
    ax[2].axis('equal')
    
    plt.show()

def calculate_accuracy(completed_paths, solution_paths, num_points=100):
    total_error = 0
    num_segments = 0
    for completed_path, solution_path in zip(completed_paths, solution_paths):
        resampled_completed = resample_path(completed_path, num_points)
        resampled_solution = resample_path(solution_path, num_points)
        for completed_segment, solution_segment in zip(resampled_completed, resampled_solution):
            error = np.linalg.norm(completed_segment - solution_segment, axis=1)
            total_error += np.sum(error)
            num_segments += len(completed_segment)
    return total_error / num_segments

input_file = "occlusion1.csv"  
solution_file = "occlusion1_sol.csv"  

paths = read_csv(input_file)
solution_paths = read_csv(solution_file)

plot_paths_comparison(paths, solution_paths, "Original Paths", "Solution Paths")

occlusions = detect_occlusions(paths)

completed_paths = complete_occlusions(paths, occlusions)

plot_paths_comparison_three(paths, solution_paths, completed_paths, "Original Paths", "Solution Paths", "Completed Paths")

accuracy = calculate_accuracy(completed_paths, solution_paths)
print("Accuracy :", accuracy)
