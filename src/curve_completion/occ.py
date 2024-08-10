import numpy as np
import matplotlib.pyplot as plt

def generate_occluded_curves():
    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.column_stack((100 + 80 * np.cos(t), 100 + 80 * np.sin(t)))
    ellipse = np.column_stack((100 + 120 * np.cos(t), 100 + 40 * np.sin(t)))
    
    ellipse_occluded = ellipse[ellipse[:, 0] > 50] 
    return [circle, ellipse_occluded]

def handle_occlusions(curves):
    completed_curves = []
    for curve in curves:
        if isinstance(curve, np.ndarray) and curve.ndim == 2 and curve.shape[1] == 2:
            completed_curve = complete_curve(curve)
            completed_curves.append(completed_curve)
        else:
            print(f"Invalid curve with shape: {curve.shape}")
    return completed_curves

def complete_curve(curve):
    return curve

def plot_paths(curves, title="Curves", color="blue", linewidth=2):
    fig, ax = plt.subplots()
    for segment in curves:
        if segment.ndim == 2:
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=linewidth)
        else:
            print(f"Skipping invalid segment with shape: {segment.shape}")
    ax.set_title(title)
    plt.show()

def main():
    curves = generate_occluded_curves()
    completed_curves = handle_occlusions(curves)
    plot_paths(completed_curves, title="Completed Curves", color="red")

if __name__ == "__main__":
    main()
