import os
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# Define activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Forward propagation function
def forward(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    output = sigmoid(z2)
    return z1, a1, z2, output


# Backward propagation function
def backward(X, y, z1, a1, z2, output, W1, W2):
    m = X.shape[0]

    # Output layer error
    output_error = output - y
    output_delta = output_error * sigmoid_derivative(output)

    # Hidden layer error
    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(a1)

    # Calculate gradients
    W2_grad = np.dot(a1.T, output_delta) / m
    b2_grad = np.sum(output_delta, axis=0, keepdims=True) / m
    W1_grad = np.dot(X.T, hidden_delta) / m
    b1_grad = np.sum(hidden_delta, axis=0, keepdims=True) / m

    return W1_grad, b1_grad, W2_grad, b2_grad


# Initialize weights
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2


# Model training function
def train(X_train, y_train, hidden_size, output_size, epochs=3000, learning_rate=0.01):
    input_size = X_train.shape[1]
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Forward propagation
        z1, a1, z2, output = forward(X_train, W1, b1, W2, b2)

        # Backward propagation
        W1_grad, b1_grad, W2_grad, b2_grad = backward(X_train, y_train, z1, a1, z2, output, W1, W2)

        # Update weights and biases
        W1 -= learning_rate * W1_grad
        b1 -= learning_rate * b1_grad
        W2 -= learning_rate * W2_grad
        b2 -= learning_rate * b2_grad

    print("Training completed")
    return W1, b1, W2, b2


# Model prediction function
def predict(X, W1, b1, W2, b2):
    _, _, _, output = forward(X, W1, b1, W2, b2)
    return np.argmax(output, axis=1), output


# Load PGM image
def load_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return image.flatten()  # Flatten the image to a 1D array


# Load data from multiple subfolders
def load_multiple_class_data(folder):
    X = []
    y = []
    classes = sorted(os.listdir(folder))  # Get all subfolders
    label_dict = {label: idx for idx, label in enumerate(classes)}
    for label in classes:
        subject_folder = os.path.join(folder, label)
        if os.path.isdir(subject_folder):
            for filename in os.listdir(subject_folder):
                if filename.endswith('.pgm'):  # Only process PGM files
                    filepath = os.path.join(subject_folder, filename)
                    X.append(load_image(filepath))
                    y.append(label_dict[label])

    X = np.array(X)
    y = np.array(y)
    y_one_hot = np.zeros((y.size, len(classes)))  # Convert labels to one-hot encoding
    y_one_hot[np.arange(y.size), y] = 1
    return X, y_one_hot, y, label_dict


# Apply PCA or LDA for preprocessing
def apply_pca_lda(X_train, X_test, y_train, method='PCA', n_components=100):
    if method == 'PCA':
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    elif method == 'LDA':
        lda = LDA(n_components=n_components)
        X_train = lda.fit_transform(X_train, np.argmax(y_train, axis=1))
        X_test = lda.transform(X_test)
    return X_train, X_test


# Set paths
train_folder = "img/faces-training"
test_folder = "img/faces-testing"

# Load all data
X_train, y_train, y_train_labels, label_dict = load_multiple_class_data(train_folder)
X_test, y_test, y_test_labels, _ = load_multiple_class_data(test_folder)

# Select PCA or LDA for preprocessing
X_train_pca, X_test_pca = apply_pca_lda(X_train, X_test, y_train, method='PCA', n_components=100)

# Train the model
W1, b1, W2, b2 = train(X_train_pca, y_train, hidden_size=100, output_size=len(label_dict), epochs=20000,
                       learning_rate=0.01)

# Test and compute recognition results for each image
correct_predictions = 0
total_predictions = 0

for i, x in enumerate(X_test_pca):
    true_label = np.argmax(y_test[i])
    predicted_label, output_probabilities = predict(x.reshape(1, -1), W1, b1, W2, b2)

    recognition_rate = output_probabilities[0, predicted_label[0]] * 100
    closest_subject = list(label_dict.keys())[list(label_dict.values()).index(predicted_label[0])]

    if predicted_label[0] == true_label:
        correct_predictions += 1
        print(
            f"Image {i + 1}: Closest to {closest_subject} with recognition rate: {recognition_rate:.2f}% - correct recognition")
    else:
        print(f"Image {i + 1}: Closest to {closest_subject} with recognition rate: {recognition_rate:.2f}%")

    total_predictions += 1

# Compute and print model accuracy
accuracy = correct_predictions / total_predictions * 100
print(f'Model accuracy: {accuracy:.2f}%')
