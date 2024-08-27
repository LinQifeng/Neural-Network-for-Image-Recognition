import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_images_from_folder(base_folder, subfolders):
    images = []
    labels = []
    for subject in subfolders:
        subject_folder = os.path.join(base_folder, subject)
        if os.path.isdir(subject_folder):
            for filename in os.listdir(subject_folder):
                if filename.endswith('.pgm'):
                    img_path = os.path.join(subject_folder, filename)
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        images.append(image.flatten())  # Flatten the image into a 1D array
                        labels.append(subject)  # Use the subject (e.g., 's1') as the label
                    else:
                        print(f"Warning: Image {img_path} could not be read.")
        else:
            print(f"Warning: Folder {subject_folder} does not exist.")
    if len(images) == 0:
        print(f"Warning: No images found in specified subfolders.")
    return np.array(images), np.array(labels)


def train_and_test_knn(training_base_folder, testing_base_folder):
    subfolders = [f"s{i}" for i in range(1, 41)]
    # Load training data
    train_images, train_labels = load_images_from_folder(training_base_folder, subfolders)

    # Check if training data is loaded correctly
    if len(train_images) == 0 or len(train_labels) == 0:
        raise ValueError("No training data found. Please check your training folder path and contents.")

    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the KNN model
    knn.fit(train_images, train_labels)

    # For each subject, test only on images from that subject
    results = {}
    for subject in subfolders:
        subject_folder = os.path.join(testing_base_folder, subject)
        if os.path.isdir(subject_folder):
            test_images, test_labels = load_images_from_folder(testing_base_folder, [subject])

            # Check if there are images in the test set
            if len(test_images) == 0:
                print(f"No test images found for {subject}. Skipping...")
                continue

            predictions = knn.predict(test_images)
            accuracy = accuracy_score(test_labels, predictions)
            results[subject] = accuracy
            print(f"Recognition accuracy for {subject}: {accuracy * 100:.2f}%")

    return results


# Define the paths for training and testing folders
training_folder = "img/faces-training"
testing_folder = "img/faces-testing"

# Run the KNN training and testing
results = train_and_test_knn(training_folder, testing_folder)
