import os

# Directory containing the files
directory = "renders/vid"

# Number of frames (assuming 600 frames as stated)
total_frames = 600

# Calculate the number of digits needed for zero-padding
num_digits = len(str(total_frames - 1))

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.startswith("frame") and filename.endswith(".png"):
        # Extract the frame number from the filename
        frame_number = int(filename[5:-4])  # Extract the number part
        
        # Create the new filename with zero-padded frame number
        new_filename = f"frame{frame_number:0{num_digits}d}.png"
        
        # Get the full paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)

print("Renaming complete!")
print([img for img in os.listdir(directory) if img.endswith(".png")])