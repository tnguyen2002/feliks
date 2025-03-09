import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
from PIL import Image, ImageOps
import math

def draw_detected_faces(image, cube_faces, color_faces):
    """
    Overlays detected Rubik's Cube faces and their colors on the original image.
    
    :param image: Original image (OpenCV BGR format).
    :param cube_faces: List of quadrilateral contours for detected faces.
    :param color_faces: List of 3x3 detected colors (RGB tuples).
    :return: Annotated image.
    """
    image_copy = image.copy()

    for face_idx, (face, colors) in enumerate(zip(cube_faces, color_faces)):
        # Convert quadrilateral points to integer format
        pts = np.array(face, dtype=np.int32)

        # Draw face outline
        cv2.polylines(image_copy, [pts], isClosed=True, color=(0, 255, 255), thickness=3)

        # Perspective transform to straighten the face
        side = 200  # Standard size for transformation
        dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
        warped = cv2.warpPerspective(image, M, (side, side))

        # Grid size
        step = side // 3

        # Draw individual stickers
        for i in range(3):
            for j in range(3):
                x, y = j * step, i * step
                color = tuple(int(c) for c in colors[i][j])  # Convert to integer BGR
                
                # Transform back to original perspective
                src_pts = np.array([
                    [x, y], [x + step, y], [x + step, y + step], [x, y + step]
                ], dtype=np.float32)
                dst_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), np.linalg.inv(M)).reshape(-1, 2)

                # Draw filled rectangles for each sticker
                cv2.fillPoly(image_copy, [np.int32(dst_pts)], color)

    return image_copy


def draw_hough_lines(img, lines):
    """Draws Hough lines on top of the original image."""
    img_with_lines = img.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines
    
    return img_with_lines


def display_image(img):
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')
    plt.show()
