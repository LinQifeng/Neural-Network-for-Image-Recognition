import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image_path = 'img/items/bicycle/2008_000036.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the custom filter (Sobel operator)
sobel_filter = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])

# Apply the filter
filtered_image = cv2.filter2D(image, -1, sobel_filter)

# Display the original image and the filtered image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Filtered Image')
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')

plt.show()
