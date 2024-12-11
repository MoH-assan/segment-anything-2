import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
import json
import cv2
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def sort_and_filter_images(directory, keyword):
    """
    Scans a directory for JPEG images, filters those that include a specific keyword in the filename,
    and sorts them based on the integer value that appears after 'f_' in the filename.

    Parameters:
    - directory (str): The path to the directory containing the images.
    - keyword (str): The keyword to filter filenames by (e.g., '_c1').

    Returns:
    - List[str]: A list of filtered and sorted filenames.
    """
    # Scan all the JPEG frame names in the directory
    frame_names = [
        p for p in os.listdir(directory)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg",'png']
    ]

    # Filter to include only images that contain the keyword
    filtered_frames = [name for name in frame_names if keyword in name]

    # Sort by the integer number after 'f_'
    filtered_frames.sort(key=lambda name: int(name.split('_f_')[1].split('_')[0]))

    return filtered_frames
def sample_points(mask, num_samples):
    if num_samples == 0:
        return np.array([[]])
    """Sample points where the mask is True."""
    # Find indices where the mask is True
    y_indices, x_indices = np.where(mask)
    total_points = len(x_indices)
    if num_samples > total_points:
        raise ValueError(f"num_samples ({num_samples}) should be less than or equal to the total number of points ({total_points}).")
    # Randomly select `num_samples` points
    chosen_indices = np.random.choice(len(x_indices), num_samples, replace=False)
    # Extract the corresponding points
    sampled_points = np.array(list(zip(x_indices[chosen_indices], y_indices[chosen_indices])), dtype=np.float32)
    return sampled_points

def total_frames_firstlight(meta_data):
    header_size = 0  # Update if there is a header size
    img_size_bytes = meta_data['height'] * meta_data['width'] * 2  # 2 bytes per pixel

    # Open the file and calculate the total file size
    with open(meta_data['filename_abs'], 'rb') as f:
        f.seek(0, 2)  # Move the cursor to the end of the file
        total_size = f.tell()  # Total size of the file

    # Calculate the total number of frames
    num_frames = (total_size - header_size) / img_size_bytes
    return int(num_frames)

def open_frame_firstlight(meta_data, index):
    header_size=0
    # Calculate the size of each image in bytes
    img_size = meta_data['height'] * meta_data['width']
    # Calculate the offset to the desired image
    
    with open(meta_data['filename_abs'], 'rb') as f:
        f.seek(0, 2)  # Move the cursor to the end of the file
        total_size = f.tell()  # `tell` gives you the current position, which is the size
        #print("Total file size using file object:", total_size, "bytes")
    offset = header_size + np.int64(img_size) * 2 * np.int64(index)  # 2 bytes per pixel for 16-bit images
    with open(meta_data['filename_abs'], 'rb') as f:
        # Seek to the desired position
        f.seek(offset)
        # Read the image data and reshape it
        img_data = np.fromfile(f, dtype=np.int16, count=img_size)  # read as 16-bit signed integers
        img = img_data.reshape(meta_data['height'], meta_data['width'])
        #print(f'relative_frame  = {index}, first four {img[0,:4]}')
        # Extract the first two pixels from the first row
        first_pixel = img[0, 0]
        second_pixel = img[0, 1]

        # Combine the first and second pixels to form a 32-bit integer frame counter
        # Convert the pixels to 32-bit integers to avoid issues with negative values and bit manipulation
        frame_counter = (second_pixel.astype(np.int32) << 16) | (first_pixel & 0xFFFF)

        #print("Frame counter:", frame_counter)
        img[0, :4] = 0    #Removing the tag 
        white_pixel = 7000
        black_pixel = 0
        img[img > white_pixel] = white_pixel
        img[img < black_pixel] = black_pixel
        img_normalized = 255 * (img.astype(np.int32)) / (white_pixel) #TODO check if this is the correct normalization to keep the details and avoid artifical oversaturation
        image_plus_camera_frame_counter = {'camera_frame_counter': frame_counter, 'img': img_normalized}
    return image_plus_camera_frame_counter

def extract_color_masks(image_path, colors_info):
    """
    Extracts specified color regions from an image based on provided color thresholds in HSV space.

    Parameters:
    - image_path (str): The file path to the image.
    - colors_info (dict): A dictionary where keys are color names and values are tuples with lower and upper HSV thresholds.

    Returns:
    - dict: A dictionary with color names as keys and the corresponding binarized masks as values.
    """
    # Load the image and convert to RGB and HSV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Dictionary to hold the color masks
    color_masks = {}
    color_masks['image'] = image_rgb
    # Process each color
    for color_name, (lower_hsv, upper_hsv) in colors_info.items():
        # Create mask for current color
        mask = cv2.inRange(image_hsv, np.array(lower_hsv), np.array(upper_hsv))
        # Store the mask in the dictionary
        color_masks[color_name] = mask

    return color_masks

def save_numpy_as_png(array, path, size_inches, dpi, cmap='gray'):
    """
    Saves a NumPy array as a PNG image with specified size and resolution.

    Parameters:
    - array (numpy.ndarray): The 2D array to save as an image.
    - path (str): File path where the image will be saved.
    - size_inches (tuple): Size of the figure in inches (width, height).
    - dpi (int): Resolution of the image in dots per inch.
    """
    # Create a figure with the specified size and DPI
    fig, ax = plt.subplots(figsize=size_inches, dpi=dpi)
    ax.imshow(array, cmap=cmap) 
    ax.axis('off')  # Turn off axis
    
    # Save the figure
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory