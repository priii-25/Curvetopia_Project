import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt

def load_data(csv_file):
    data = pd.read_csv(csv_file, header=None)
    x_coords = data[2].values
    y_coords = data[3].values
    return np.vstack((x_coords, y_coords)).T

def detect_intersections(curve):
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
                    intersections.append((i, intersection_point))
                    interrupted_segments.append((i, j)) 
    
    return intersections, interrupted_segments

def extract_figures(curve, intersections):
    figures = []
    last_idx = 0
    
    for idx, intersection_point in intersections:
        figure = curve[last_idx:idx+1]
        figures.append(figure)
        last_idx = idx
    
    # Add the last segment
    figures.append(curve[last_idx:])
    
    return figures

def main():
    csv_file = 'src/curve_completion/occlusion1.csv'  
    curve = load_data(csv_file)
    
    intersections, interrupted_segments = detect_intersections(curve)
    
    if intersections:
        figures = extract_figures(curve, intersections)
        print(f"Extracted {len(figures)} figures.")
        
        for i, figure in enumerate(figures):
            plt.plot(figure[:, 0], figure[:, 1], label=f'Figure {i+1}')
        
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Extracted Figures from Curve')
        plt.show()

    else:
        print("No intersections detected. The entire curve is one figure.")
        plt.plot(curve[:, 0], curve[:, 1], 'bo-', label='Single Figure')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Single Figure from Curve')
        plt.show()

if __name__ == "__main__":
    main()

