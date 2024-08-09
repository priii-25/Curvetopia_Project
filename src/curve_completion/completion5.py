import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, MultiLineString
from shapely.validation import explain_validity

def load_data(csv_file):
    data = pd.read_csv(csv_file, header=None)
    x_coords = data[2].values
    y_coords = data[3].values
    return np.vstack((x_coords, y_coords)).T

def validate_geometry(geometry):
    if not geometry.is_valid:
        print(f"Invalid geometry: {explain_validity(geometry)}")
        geometry = geometry.buffer(0)
        if not geometry.is_valid:
            raise ValueError("Geometry is still invalid after buffering.")
    return geometry

def identify_occlusion(curve, complex_figure):
    line = LineString(curve)
    complex_poly = Polygon(complex_figure)
    
    line = validate_geometry(line)
    complex_poly = validate_geometry(complex_poly)
    
    intersection = line.intersection(complex_poly)
    return intersection

def clip_shape(occlusion, complex_figure):
    clipped_lines = []
    complex_poly = Polygon(complex_figure)
    complex_poly = validate_geometry(complex_poly)

    if occlusion.geom_type == 'LineString':
        clipped_line = occlusion.intersection(complex_poly)
        if not clipped_line.is_empty:
            clipped_lines.append(clipped_line)
    elif occlusion.geom_type == 'MultiLineString':
        for line in occlusion.geoms:
            clipped_line = line.intersection(complex_poly)
            if not clipped_line.is_empty:
                clipped_lines.append(clipped_line)
    
    return MultiLineString(clipped_lines) if clipped_lines else None

def extract_occluded_shape(curve, complex_figure):
    occlusion = identify_occlusion(curve, complex_figure)
    if not occlusion.is_empty:
        return clip_shape(occlusion, complex_figure)
    return None

def main():
    csv_file = 'src/curve_completion/occlusion1.csv'
    complex_figure = load_data(csv_file)
    
    curve = complex_figure[:100]
    
    occluded_shape = extract_occluded_shape(curve, complex_figure)
    
    plt.plot(complex_figure[:, 0], complex_figure[:, 1], 'bo-', label='Complex Figure')
    if occluded_shape:
        if occluded_shape.geom_type == 'MultiLineString':
            for line in occluded_shape.geoms:
                x, y = line.xy
                plt.plot(x, y, 'r-', label='Occluded Shape')
        elif occluded_shape.geom_type == 'LineString':
            x, y = occluded_shape.xy
            plt.plot(x, y, 'r-', label='Occluded Shape')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Occlusion Detection and Extraction')
    plt.show()

if __name__ == "__main__":
    main()
