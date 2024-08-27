import os
import random
import shutil


def split_images(source_folder, training_folder, testing_folder, num_to_select):
    subjects = [f"s{i}" for i in range(1, 41)]  # s1 to s40

    for subject in subjects:
        source_path = os.path.join(source_folder, subject)
        training_path = os.path.join(training_folder, subject)
        testing_path = os.path.join(testing_folder, subject)

        # Create training and testing folders
        os.makedirs(training_path, exist_ok=True)
        os.makedirs(testing_path, exist_ok=True)

        # Get all image files in the current subject folder
        images = os.listdir(source_path)

        # Randomly select the specified number of images for the training set
        training_images = random.sample(images, num_to_select)

        for image in images:
            # Determine the source path of the image
            image_path = os.path.join(source_path, image)
            # Set the destination folder based on whether the image is in the training set
            if image in training_images:
                destination = os.path.join(training_path, image)
            else:
                destination = os.path.join(testing_path, image)

            # Copy the image to the destination path
            shutil.copy(image_path, destination)


# Define folder paths
source_folder = "img/faces"
training_folder = "img/faces-training"
testing_folder = "img/faces-testing"

# Call the function to split the images
split_images(source_folder, training_folder, testing_folder, 8)
