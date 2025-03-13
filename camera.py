import cv2
import os
from vision import process_color_images
from utils import colorToLetter
import twophase.solver  as sv

def capture_images():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    save_directory = "images/scramble_3"
    os.makedirs(save_directory, exist_ok=True)
    
    instructions = {
        "yellow": "Yellow Face with Blue on top",
        "orange": "Orange face with yellow on top",
        "blue": "Blue face with yellow on top",
        "red": "Red face with yellow on top",
        "green": "Green face with yellow on top",
        "white": "White with green on top"
    }
    
    print("Press SPACE to capture an image. Press ESC to exit.")
    
    image_names = list(instructions.keys())
    img_count = 0
    
    while img_count < 6:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Determine cropping region
        height, width, _ = frame.shape
        square_size = min(width, height) // 1
        top_left_x = (width - square_size) // 2
        top_left_y = (height - square_size) // 2
        bottom_right_x = (width + square_size) // 2
        bottom_right_y = (height + square_size) // 2
        
        # Crop the region inside the square
        cropped_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        
        # Save the cropped image
        img_filename = os.path.join(save_directory, f"{image_names[img_count]}.jpg")
        cv2.imwrite(img_filename, cropped_frame)
        # print(f"Saved {img_filename}")
        
        # Display instruction on the camera feed
        instruction_text = instructions[image_names[img_count]]
        cv2.putText(frame, instruction_text, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Draw a square outline in the center of the frame
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
        
        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # SPACE key
            img_count += 1
        elif key == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Capture complete.")

if __name__ == "__main__":
    capture_images()