import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
from PIL import Image, ImageOps
import math
from utils import getColors, cv2_display_image
from rembg import remove
import os


# RUBIKS_COLORS = getColors()


def get_hough_lines(edges):
    """Detects lines using Hough Transform."""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=10)
    return lines

def canny_edge_detector(img):
     edges = cv2.Canny(img, 0, 150)
     return edges

def draw_hough_lines(img, lines):
    """Draws Hough lines on top of the original image."""
    black_image = np.zeros_like(img)

    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(black_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines
    
    return black_image

def find_corners(image):
    target_color = np.array([0, 255, 0])  # Example: Pure Red

    # Find all pixels matching the target color
    matches = np.all(image == target_color, axis=-1)

    # Get the coordinates of matching pixels
    coordinates = np.column_stack(np.where(matches))  # (y, x) format

    # Find the top-left and bottom-right coordinates
    top_left = tuple(coordinates.min(axis=0)[::-1])  # Swap (y, x) -> (x, y)
    bottom_right = tuple(coordinates.max(axis=0)[::-1])

    # print(f"Top-left most pixel: {top_left}")
    # print(f"Bottom-right most pixel: {bottom_right}")
    return top_left, bottom_right



def kmeans_color(img, k=1):
    """Apply KMeans clustering to find the dominant color in an image region."""
    pixels = img.reshape(-1, 3)  # Flatten the region into a 2D array
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_.astype(int)  # Get cluster center (average color)
    return dominant_color[0]  # Return the dominant color

def closest_rubiks_color(color, rubiks_colors):
    """Find the closest Rubik's Cube color based on the minimum Euclidean distance."""
    min_distance = float("inf")
    closest_color = None

    for name, rgb_list in rubiks_colors.items():
        for rgb in rgb_list:  # Compare against all stored RGB values
            distance = np.linalg.norm(np.array(color) - np.array(rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = name

    return closest_color

def classify_color(image, i, j, RUBIKS_COLORS):
    # Convert to HSV
    median_rgb = np.median(image, axis=(0, 1))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Compute the average color
    median_hsv = np.median(image, axis=(0, 1)) # median instead
    print("i, j", i , j)
    h, s, v = median_hsv
    print(h,s,v)
    
    if s <= 85 and v >= 140:  # Low saturation, high brightness
        return "white"
    elif ((0 <= h <= 8) or (160 <= h <= 180)) :
        return "red"
    elif 8 < h <= 20:
        return "orange"
    elif 21 <= h <= 35:
        return "yellow"
    elif 42 <= h <= 72.5: 
        return "green"
    elif 85 <= h <= 135:  
        return "blue"
    else:
        return closest_rubiks_color(median_rgb, RUBIKS_COLORS)

def split_and_kmeans(image, top_left, bottom_right, RUBIKS_COLORS):
    """Splits a region of an image into a 3x3 grid and finds the closest Rubik's Cube color for each region."""
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Extract region
    region = image[y1:y2, x1:x2]
    h, w, _ = region.shape

    # Determine grid size
    grid_h, grid_w = h // 3, w // 3

    # Store colors

    color_names = []

    # Process each 3x3 subregion
    for i in range(3):
        row_colors = []
        row_names = []
        for j in range(3):
            x_start, y_start = j * grid_w, i * grid_h
            x_end, y_end = x_start + grid_w, y_start + grid_h

            subregion = region[y_start:y_end, x_start:x_end]
            # dominant_color = kmeans_color(subregion)
            closest_color = classify_color(subregion, i , j, RUBIKS_COLORS)
            # closest_color = classify_color(subregion)
            row_names.append(closest_color)  # Save color name
        
        color_names.append(row_names)
    return color_names


def imageToColors(image_path):
    image = cv2.imread(image_path)
    # image = remove(image)
    image = image[:, :, :3] 
    edges = canny_edge_detector(image)
    # cv2_display_image(edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=30, maxLineGap=10)
    hough_line_image = draw_hough_lines(image, lines)
    top_left, bottom_right = find_corners(hough_line_image)
    color_names = split_and_kmeans(image, top_left, bottom_right, {})
    # print(color_names)
    cv2_display_image(image)
    return color_names


def process_color_images(directory_path):
    """
    Processes all images in a directory and returns a dictionary where:
    - The keys are the colors extracted from filenames (e.g., "red" from "red.png").
    - The values are the detected colors from the image.

    Args:
        directory_path (str): Path to the directory containing images.

    Returns:
        dict: A dictionary mapping color names to detected face colors.
    """
    color_faces = {}
    corners = []
    RUBIKS_COLORS = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):  # Ensure only PNG files are processed
        color_name = filename.split(".")[0] # Extract the name before .png
        # detected_colors = imageToColors(directory_path + filename)  # Process the image
        image_path = directory_path + filename
        image = cv2.imread(image_path)
        image = image[:, :, :3] 
        edges = canny_edge_detector(image)
        # cv2_display_image(edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=30, maxLineGap=10)
        hough_line_image = draw_hough_lines(image, lines)
        top_left, bottom_right = find_corners(hough_line_image)
        x1, y1 = top_left
        x2, y2 = bottom_right
        corners.append((top_left, bottom_right))
        
        # Extract region
        region = image[y1:y2, x1:x2]
        h, w, _ = region.shape

        # Determine grid size
        grid_h, grid_w = h // 3, w // 3

        # Process each 3x3 subregion

        x_start, y_start = grid_w, grid_h
        x_end, y_end = x_start + grid_w, y_start + grid_h

        center = region[y_start:y_end, x_start:x_end]
        # image = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        # Compute the average color
        median_rgb = np.median(center, axis=(0, 1)) 
        RUBIKS_COLORS[color_name] = median_rgb
    # print("RUBIKS COLORS", RUBIKS_COLORS)

    for i,filename in enumerate(os.listdir(directory_path)):  # Ensure only PNG files are processed
        color_name = filename.split(".")[0] # Extract the name before .png
        top_left, bottom_right = corners[i]
        image_path = directory_path + filename
        image = cv2.imread(image_path)
        image = image[:, :, :3] 
        color_names = split_and_kmeans(image, top_left, bottom_right, RUBIKS_COLORS)
        color_faces[color_name] = color_names
    return color_faces
