import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ciede2000 import CIEDE2000

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

# From https://stackoverflow.com/questions/13405956/convert-an-image-rgb-lab-with-python/16020102#16020102
def rgb2lab ( inputColor ) :

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab

#receiving bgr colors
def closest_rubiks_color(median_bgr, RUBIKS_COLORS):
    b, g, r = median_bgr
    min_distance = float("inf")
    closest_color = None
    

    for name, color in RUBIKS_COLORS.items():
        
        distance = CIEDE2000(rgb2lab((r,g,b)), rgb2lab((color[2], color[1], color[0])))
        if distance < min_distance:
            min_distance = distance
            closest_color = name

    return closest_color

def closest_rubiks_color_hsv_range(median_hsv, RUBIKS_COLORS):
    h, s, v = median_hsv
    min_distance = float("inf")
    closest_color = None

    for name, val in RUBIKS_COLORS.items():
        distance = CIEDE2000(median_hsv, val)
        if distance < min_distance:
            min_distance = distance
            closest_color = name

    return closest_color

def classify_color(median_hsv, median_bgr, median_lab, i, j, RUBIKS_COLORS):
    closest_color = closest_rubiks_color(median_bgr, RUBIKS_COLORS)
    return closest_color
    

def getColors(base_path="images/colors"):
    RUBIKS_COLORS = {}

    for color in os.listdir(base_path):
        color_path = os.path.join(base_path, color)
        if os.path.isdir(color_path):
            color_values = []
            
            for file in os.listdir(color_path):
                img_path = os.path.join(color_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
                    avg_hsv= img.mean(axis=(0, 1)) 
                    color_values.append(avg_hsv)  

            if color_values:
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

def sortAllSquares(allSquares):
    allSquares_copy = allSquares[:]
    allSquares_copy.sort(key=lambda sq: sq.centroid[1])  

    rows = [allSquares_copy[i:i+3] for i in range(0, 9, 3)]

    for row in rows:
        row.sort(key=lambda sq: sq.centroid[0])
    return rows


def classify_color_weighted(hsv_value, RUBIKS_COLORS):
    weights = np.array([3.0, 1.0, 1.0])
    min_dist = float('inf')
    closest_color = None
    for color, center in RUBIKS_COLORS.items():
        diff = hsv_value - center
        weighted_diff = diff * weights
        dist = np.linalg.norm(weighted_diff)
        if dist < min_dist:
            min_dist = dist
            closest_color = color
    return closest_color


def colorsFromSquares(allSquares):
    allSquares_sorted = [] 
    for i in range(len(allSquares)):
        allSquares_sorted.append(sortAllSquares(allSquares[i]))
    face = {}
    RUBIKS_COLORS = {}
 

    for i in range(len(allSquares_sorted)):
        squares = allSquares_sorted[i]
        if(len(squares) > 0):
            center = squares[1][1].center
            center_hsv = squares[1][1].color_hsv
            center_bgr = squares[1][1].color_bgr
            center_lab = squares[1][1].color_lab
            RUBIKS_COLORS[center] = center_bgr
    print("Rubiks Colors", RUBIKS_COLORS)

    for i in range(len(allSquares_sorted)):
        squares = allSquares_sorted[i]
        if(len(squares) > 0):
            center = squares[1][1].center
            face[center] = []
            for j in range(3):
                row = []
                for k in range(3):
                    square = squares[j][k]

                    color_hsv = square.color_hsv
                    color_bgr = square.color_bgr
                    color_lab = square.color_lab

                    classified_color= classify_color(color_hsv, color_bgr, color_lab, j, k, RUBIKS_COLORS)
                    
                    if(j == 1 and k == 1):
                        classified_color = center
                        
                    row.append(classified_color)
                face[center].append(row)

    return face




    