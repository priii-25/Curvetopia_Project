import cv2
import numpy as np
from skimage import measure
from skimage.filters import gaussian

def enhance_contrast(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

def apply_gaussian_filter(image, sigma=1):
    return gaussian(image, sigma=sigma)

def adjust_intensity(image):
    adjusted = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return adjusted

def detect_shapes(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        shape_factor = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        num_vertices = len(approx)
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        if num_vertices == 3 or (0.4 <= shape_factor <= 0.7):
            shape = "Triangle"
        elif num_vertices == 4:
            if 0.95 <= shape_factor <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif num_vertices > 4 or shape_factor > 0.8:
            shape = "Circle"
        else:
            shape = "Unknown"
        
        print(f"Shape factor: {shape_factor}, Vertices: {num_vertices}")
        shapes.append((shape, (cX, cY)))
    
    return shapes, binary

image_path = "src\shape_detection\download.png"
try:
    detected_shapes, binary_image = detect_shapes(image_path)

    for shape, centroid in detected_shapes:
        print(f"Detected {shape} at coordinates: {centroid}")

    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"An error occurred: {e}")