import cv2
import numpy as np

def display_string_cv2(message):
    # Create a white image
    img = np.ones((900, 900, 3), dtype=np.uint8) * 255

    # Define font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Get text size
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]

    # Calculate coordinates to center the text
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2

    # Put the text on the image
    cv2.putText(
        img, 
        message, 
        (text_x, text_y), 
        font, 
        font_scale, 
        (0, 0, 0),  # Black color
        thickness
    )

    # Display the image
    cv2.imshow("String Display", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
