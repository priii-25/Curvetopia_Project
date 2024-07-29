import cv2
import numpy as np
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def apply_gaussian_filter(image, sigma=1.0):
    return cv2.GaussianBlur(image, (5, 5), sigma)

def convert_to_binary(image, threshold=127):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def label_components(binary_image):
    num_labels, labels_im = cv2.connectedComponents(binary_image)
    return num_labels, labels_im

def calculate_shape_factor(region, labeled_image):
    area = cv2.contourArea(region)
    perimeter = cv2.arcLength(region, True)
    if perimeter == 0:
        return None
    shape_factor = (4 * np.pi * area) / (perimeter ** 2)
    
    if 0.7 <= shape_factor <= 0.8:
        return 'Circle'
    elif 0.484 <= shape_factor <= 0.55:
        return 'Square'
    elif 0.2 <= shape_factor <= 0.3:
        return 'Rectangle'
    elif 0.44 <= shape_factor <= 0.483:
        return 'Triangle'
    elif 0.32 <= shape_factor <= 0.34:
        return 'Oval'
    elif 0.36 <= shape_factor <= 0.38:
        return 'Diamond'
    else:
        return 'Unknown'

def recognize_shapes(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = apply_clahe(gray_image)
    filtered_image = apply_gaussian_filter(enhanced_image)
    adjusted_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    binary_image = convert_to_binary(adjusted_image)
    
    num_labels, labeled_image = label_components(binary_image)
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    recognized_shapes = []
    for contour in contours:
        shape = calculate_shape_factor(contour, labeled_image)
        if shape:
            recognized_shapes.append(shape)
    
    return recognized_shapes

image_path = 'src\shape_detection\myCircle.png'
recognized_shapes = recognize_shapes(image_path)
print("Recognized Shapes:", recognized_shapes)
