# Curve Analysis and Shape Detection

## Overview

This project provides tools for analyzing and fitting curves to different geometric shapes. It includes functionalities for detecting intersections in curves, completing interrupted segments, fitting shapes, and visualizing the results. The project also supports saving results in both SVG and CSV formats.

## Features

- **Curve Loading**: Load curve data from CSV files.
- **Intersection Detection**: Identify and handle intersections and occlusions in curves.
- **Curve Completion**: Extend interrupted segments of curves.
- **Shape Detection**: Determine if a curve resembles a circle, ellipse, or line.
- **Shape Fitting**: Fit geometric shapes to curves and generate points for visualization.
- **Visualization**: Plot original and fitted curves using `matplotlib`.
- **Output**: Save results as SVG and CSV files.

## Installation

To install the required dependencies, use the following command:

```bash
pip install numpy matplotlib shapely opencv-python svgwrite
