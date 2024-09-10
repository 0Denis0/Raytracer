# genVideo

import cv2
import os
import subprocess

def genVideo(imageFolder, videoName):
    print('start')
    image_folder = imageFolder
    video_name = videoName

    # List all the images in the directory and sort them
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    # Read the first image to get the frame properties
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # Set the desired width and height
    width, height = 1920, 1080

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

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

def mov_to_gif(input_mov, output_gif, ffmpeg_location="C:/ffmpeg/bin/ffmpeg.exe", fps=10, scale=-1):
    """
    Convert a .mov video file to a .gif using ffmpeg.

    :param input_mov: The path to the input .mov file.
    :param output_gif: The desired path to the output .gif file.
    :param ffmpeg_location: The path to the ffmpeg executable.
    :param fps: Frames per second for the GIF (default 10).
    :param scale: Scaling value for the gif, -1 preserves aspect ratio (default -1).
    """
    # Build the ffmpeg command
    ffmpeg_command = [
        ffmpeg_location,           # Path to ffmpeg executable
        '-i', input_mov,           # Input file
        '-vf', f'fps={fps},scale={scale}:-1:flags=lanczos', # Video filter: frames per second, scale
        '-y', output_gif           # Output file, overwrite without asking
    ]
    
    # Run the ffmpeg command using subprocess
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Conversion successful: {output_gif}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

def pngs_to_gif(png_folder, output_gif, ffmpeg_location="C:/ffmpeg/bin/ffmpeg.exe", fps=10, scale=-1):
    """
    Convert a folder of .png images to a .gif using ffmpeg.
    
    :param png_folder: Path to the folder containing .png images.
    :param output_gif: The desired path to the output .gif file.
    :param ffmpeg_location: The path to the ffmpeg executable.
    :param fps: Frames per second for the GIF (default 10).
    :param scale: Scaling value for the gif, -1 preserves aspect ratio (default -1).
    """
    # Ensure the input folder exists
    if not os.path.exists(png_folder):
        print(f"Error: The folder '{png_folder}' does not exist.")
        return

    # Ensure the folder contains .png files
    png_files = [f for f in sorted(os.listdir(png_folder)) if f.endswith('.png')]
    if not png_files:
        print(f"Error: No .png files found in folder '{png_folder}'.")
        return

    # Create an input pattern for ffmpeg (assumes sequential naming like img001.png, img002.png, ...)
    input_pattern = os.path.join(png_folder, 'frame%04d.png')

    # Build the ffmpeg command
    ffmpeg_command = [
        ffmpeg_location,                 # Path to ffmpeg executable
        '-framerate', str(fps),          # Frames per second
        '-i', input_pattern,             # Input pattern for PNG files
        '-vf', f'scale={scale}:-1:flags=lanczos',  # Scaling (preserves aspect ratio)
        '-y', output_gif                 # Output GIF file (overwrite)
    ]

    # Run the ffmpeg command using subprocess
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"GIF created successfully: {output_gif}")
    except subprocess.CalledProcessError as e:
        print(f"Error during GIF creation: {e}")


if __name__ == "__main__":
    # # Example usage
    # input_mov_path = 'renders/vid5/motion5.mov'
    # output_gif_path = 'renders/vid5/motionFromMov.gif'
    # mov_to_gif(input_mov_path, output_gif_path, fps = 30)

    # # Example usage
    # png_folder_path = 'renders\\vid5'
    # output_gif_path = 'renders/vid5/motionFromPngs.gif'
    # pngs_to_gif(png_folder_path, output_gif_path, fps=30)

    i = 1
    folder = "renders/vid5/"
    frameNum = f"{i:04d}"
    fileName = os.path.join(folder, f"frame{frameNum}.png")
    if os.path.exists(fileName):
        print(f"{fileName} exists in {folder}")
