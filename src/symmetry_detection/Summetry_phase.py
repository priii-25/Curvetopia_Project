import numpy as np
import cv2
import svgwrite
import cairosvg
from scipy import signal
import matplotlib.pyplot as plt
from scipy import signal
from collections import defaultdict


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
    cairosvg.svg2png(url=svg_path, write_to=png_path,
                     parent_width=W, parent_height=H,
                     output_width=fact*W, output_height=fact*H,
                     background_color='white')


def create_log_gabor_filter(size, wavelength, orientation, bandwidth):
    rows, cols = size
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    center = (cols // 2, rows // 2)
    radius = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    theta = np.arctan2(y - center[1], x - center[0])
    
    orientation = np.deg2rad(orientation)
    
    angular_filter = np.exp(-((theta - orientation)**2) / (2 * bandwidth**2))
    radial_filter = np.exp(-(np.log(radius / wavelength)**2) / (2 * np.log(bandwidth)**2))
    
    gabor_filter = radial_filter * angular_filter
    
    return gabor_filter

def compute_local_energy(image, filters):
    energy = np.zeros((image.shape[0], image.shape[1], len(filters)))
    for i, filt in enumerate(filters):
        even = signal.convolve2d(image, np.real(filt), mode='same', boundary='symm')
        odd = signal.convolve2d(image, np.imag(filt), mode='same', boundary='symm')
        energy[:,:,i] = np.sqrt(even**2 + odd**2)
    return energy

def detect_symmetry_psd(curve):
    x_min, y_min = np.min(curve, axis=0)
    x_max, y_max = np.max(curve, axis=0)
    width = int(x_max - x_min) + 1
    height = int(y_max - y_min) + 1
    image = np.zeros((height, width), dtype=np.uint8)
    curve_normalized = curve - [x_min, y_min]
    cv2.polylines(image, [curve_normalized.astype(np.int32)], isClosed=True, color=255, thickness=1)
    
    scales = [2, 4, 8, 16]
    orientations = [0, 45, 90, 135]
    filters = [create_log_gabor_filter(image.shape, s, o, 0.65) for s in scales for o in orientations]
    
    energy = compute_local_energy(image, filters)
    
    epsilon = 1e-10
    total_energy = np.sum(energy, axis=2)
    total_amplitude = np.sum(np.abs(energy), axis=2)
    phase_congruency = total_energy / (total_amplitude + epsilon)
    
    symmetry_map = phase_congruency > np.mean(phase_congruency)
    y, x = np.nonzero(symmetry_map)
    if len(x) > 0 and len(y) > 0:
        center_x, center_y = np.mean(x), np.mean(y)
        dx = x - center_x
        dy = y - center_y
        orientation = np.arctan2(np.sum(dx * dy), np.sum(dx**2 - dy**2)) / 2
    else:
        orientation = 0

    symmetry_score = np.mean(phase_congruency)
    
    return orientation, symmetry_score

def detect_symmetry_in_paths(paths):
    symmetry_info = []
    for path in paths:
        curve = np.vstack(path)
        angle, score = detect_symmetry_psd(curve)
        symmetry_info.append((angle, score))
    return symmetry_info

def count_unique_symmetry_lines(symmetry_info, tolerance=1e-2):
    angle_groups = defaultdict(list)
    for angle, score in symmetry_info:
        matched = False
        for key in angle_groups:
            if np.abs(angle - key) < tolerance:
                angle_groups[key].append((angle, score))
                matched = True
                break
        if not matched:
            angle_groups[angle].append((angle, score))
    
    return len(angle_groups)

if __name__ == "__main__":
    path_XYs = read_csv("Test_cases/isolated.csv")
    symmetry_results = detect_symmetry_in_paths(path_XYs)
    
    for i, (angle, score) in enumerate(symmetry_results):
        print(f"Path {i}:")
        print(f"  Symmetry axis angle: {np.degrees(angle):.2f} degrees")
        print(f"  Symmetry score: {score:.4f} (higher is more symmetric)")
        print()
    
    unique_symmetry_lines = count_unique_symmetry_lines(symmetry_results)
    print(f"Number of unique symmetry lines: {unique_symmetry_lines}")

    plot(path_XYs)
