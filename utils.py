import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
from PIL import Image, ImageOps
import math
import os

def cv2_display_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def display_image(img):
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')
    plt.show()




def getColors(base_path="images/colors"):
    RUBIKS_COLORS = {}

    # Iterate through each subdirectory (color category)
    for color in os.listdir(base_path):
        color_path = os.path.join(base_path, color)
        if os.path.isdir(color_path):  # Ensure it's a directory
            color_values = []
            
            # Read all images in the subdirectory
            for file in os.listdir(color_path):
                img_path = os.path.join(color_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
                    avg_hsv= img.mean(axis=(0, 1)) 
                    color_values.append(avg_hsv)  

            if color_values:  # Ensure there are images
                RUBIKS_COLORS[color] = color_values 

    return RUBIKS_COLORS

def colorToLetter():
    colorToLetter = {
        "yellow" : "U",
        "green" : "F",
        "orange": "R",
        "blue": "B",
        "red": "L",
        "white": "D"
    }
    return colorToLetter

def fixColors(colors):
    for key in colors:
        colors[key][1][1] = key
    return colors