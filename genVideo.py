# genVideo

import cv2
import os

image_folder = 'renders'
video_name = 'video.avi'

# List all the images in the directory and sort them
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()

# Read the first image to get the frame properties
frame = cv2.imread(os.path.join(image_folder, images[0]))

# Set the desired width and height
width, height = 480, 270

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 5, (width, height))

# Iterate over all the images
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    if frame is not None:
        resized_frame = cv2.resize(frame, (width, height))
        video.write(resized_frame)

# Release everything
cv2.destroyAllWindows()
video.release()
