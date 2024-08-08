import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def apply_gaussian_filter(image, sigma=1.0):
    return cv2.GaussianBlur(image, (5, 5), sigma)

def convert_to_binary(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def recognize_shape(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0 or area == 0:
        return 'Unknown'
    
    shape_factor = (4 * np.pi * area) / (perimeter ** 2)
    
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    
    rect_area = w * h
    extent = float(area) / rect_area
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area
    
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    if 0.7 <= shape_factor <= 0.8 and 0.9 <= aspect_ratio <= 1.1:
        return 'Circle'
    elif 0.484 <= shape_factor <= 0.55 and 0.9 <= aspect_ratio <= 1.1 and extent > 0.8:
        return 'Square'
    elif 0.2 <= shape_factor <= 0.3 and aspect_ratio > 1.2:
        return 'Rectangle'
    elif 0.44 <= shape_factor <= 0.483 and 0.8 <= aspect_ratio <= 1.2:
        return 'Triangle'
    elif 0.32 <= shape_factor <= 0.34 and aspect_ratio > 1.1:
        return 'Oval'
    elif 0.36 <= shape_factor <= 0.38 and 0.8 <= aspect_ratio <= 1.2:
        return 'Diamond'
    else:
        return 'Unknown'

def recognize_shapes(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = apply_clahe(gray_image)
    filtered_image = apply_gaussian_filter(enhanced_image)
    edges = cv2.Canny(filtered_image, 50, 150)
    binary_image = convert_to_binary(filtered_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    recognized_shapes = []
    for contour in contours:
        shape = recognize_shape(contour)
        recognized_shapes.append(shape)
    
    return recognized_shapes

image_path = 'src\shape_detection\myCircle.png'
recognized_shapes = recognize_shapes(image_path)
print("Recognized Shapes:", recognized_shapes)

image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
enhanced_image = apply_clahe(gray_image)
filtered_image = apply_gaussian_filter(enhanced_image)
edges = cv2.Canny(filtered_image, 50, 150)
binary_image = convert_to_binary(filtered_image)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Contours')
plt.show()

