import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_edges(image):
    edges = cv2.Canny(image, 100, 200) 
    return edges
def extract_features(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
def classify_shapes(contours):
    shapes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            shapes.append("Triangle")
        elif len(approx) == 4:
            shapes.append("Quadrilateral")
        elif len(approx) > 4:
            shapes.append("Circle")
        else:
            shapes.append("Unknown")
    return shapes
def shape_recognition(image):
    preprocessed_image = preprocess_image(image)
    edges = detect_edges(preprocessed_image)
    contours = extract_features(edges)
    shapes = classify_shapes(contours)
    return shapes

image = cv2.imread('src\shape_detection\RectangleHand.webp') 
shapes = shape_recognition(image)

for shape in shapes:
    print(shape)
