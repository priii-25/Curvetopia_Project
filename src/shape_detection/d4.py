import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import csv

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_edges(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.sqrt(sobelx**2 + sobely**2)
    sobel = cv2.convertScaleAbs(sobel)
    return sobel

def post_process_edges(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def extract_features(refined_edges, num_features=625):
    contours, _ = cv2.findContours(refined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for cnt in contours:
        hu_moments = cv2.HuMoments(cv2.moments(cnt)).flatten()
        hu_moments = np.pad(hu_moments, (0, num_features - len(hu_moments)), 'constant')  # Pad to match num_features
        features.append((cnt, hu_moments))
    return features

def save_training_data(images, labels, csv_filename, num_features=625):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for image, label in zip(images, labels):
            preprocessed_image = preprocess_image(image)
            edges = detect_edges(preprocessed_image)
            refined_edges = post_process_edges(edges)
            contours, _ = cv2.findContours(refined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                hu_moments = cv2.HuMoments(cv2.moments(cnt)).flatten()
                hu_moments = np.pad(hu_moments, (0, num_features - len(hu_moments)), 'constant')  # Pad to match num_features
                writer.writerow(np.append(hu_moments, label))

def train_classifier_from_csv(csv_filename):
    data = pd.read_csv(csv_filename)
    data = data.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, coerce errors to NaN
    data = data.dropna()  # Drop rows with NaN values
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels
    y = y.astype(int)  # Ensure labels are integers
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

def classify_shapes(features, classifier):
    shapes = []
    for cnt, hu_moments in features:
        label = classifier.predict([hu_moments])[0]
        shapes.append((cnt, label))
    return shapes

def shape_recognition(image, classifier, num_features=625):
    preprocessed_image = preprocess_image(image)
    edges = detect_edges(preprocessed_image)
    refined_edges = post_process_edges(edges)
    features = extract_features(refined_edges, num_features=num_features)
    shapes = classify_shapes(features, classifier)
    return shapes

# Train classifier
classifier = train_classifier_from_csv('src/shape_detection/ShapesDataset.csv')

# Example Usage for Shape Recognition
image = cv2.imread('src\shape_detection\handDrawnTriangle.png')  # Replace with your image path
shapes = shape_recognition(image, classifier, num_features=625)

# Display shapes on the image
label_names = {0: "Circle", 1: "Square", 2: "Triangle"}
for cnt, label in shapes:
    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(image, label_names[label], (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        cX, cY = 0, 0

cv2.imshow("Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
