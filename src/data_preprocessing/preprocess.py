import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import os
import ctypes

cairo_bin_path = r'D:\msys64\mingw64\bin'
os.environ['PATH'] = cairo_bin_path + ';' + os.environ['PATH']

try:
    cairo_lib = ctypes.CDLL(r'D:\msys64\mingw64\bin\libcairo-2.dll')
    print("Cairo library loaded successfully.")
except OSError as e:
    print(f"Failed to load Cairo library: {e}")

import cairosvg

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

def plot(path_XYs, colours=['r', 'g', 'b', 'c', 'm', 'y', 'k']):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()

def polylines2svg(paths_XYs, svg_path, colours=['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    
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