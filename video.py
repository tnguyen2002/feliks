import cv2
import numpy as np
from utils import colorsFromSquares, colorToLetter, cv2_display_image
import twophase.solver  as sv
from visualizer import display_string_cv2


class Square:
    def __init__(self, contour, centroid, color_bgr, color_hsv, color_lab, center):
        self.contour = contour
        self.centroid = centroid  
        self.color_bgr = color_bgr
        self.color_hsv = color_hsv
        self.color_lab = color_lab
        self.center = center

class Camera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.colors_locked = False  
        self.locked_squares = []    
    def reset(self):
        self.colors_locked = False
        self.locked_squares = []

    def angle_between_vectors(self, v1, v2):
        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
        if magnitude == 0:
            return 0
        angle = np.arccos(dot_product / magnitude)
        return np.degrees(angle)

    def get_median_color(self, frame, contour):
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        contour_points = contour.reshape(-1, 2)

        hsv_values = []
        bgr_values = []
        lab_values = []

        for (x, y) in contour_points:
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]: 
                hsv_values.append(frame_hsv[y, x])
                bgr_values.append(frame[y, x])
                lab_values.append(frame_lab[y, x])

        if len(hsv_values) == 0:
            return (0, 0, 0), (0, 0, 0), (0, 0, 0)

        hsv_values = np.array(hsv_values)
        bgr_values = np.array(bgr_values)
        lab_values = np.array(lab_values)

        median_hsv = np.median(hsv_values, axis=(0))
        median_bgr = np.median(bgr_values, axis=(0))
        median_lab = np.median(lab_values, axis=(0))

        return median_hsv, median_bgr, median_lab
    
    def display_face(self, frame, face, grid_start_x, grid_start_y):
        cell_size = 50  
        gap = 5 

        for i, square in enumerate(face):
            row = i // 3
            col = i % 3
            
            x = grid_start_x + col * (cell_size + gap)
            y = grid_start_y + row * (cell_size + gap)
            
            color = square.color_bgr
            cv2.rectangle(frame, (x, y), (x + cell_size, y + cell_size), color, -1)
            cv2.rectangle(frame, (x, y), (x + cell_size, y + cell_size), (0, 0, 0), 1)


    def detect_squares(self, frame, squares, i, center):
        grid_start_x, grid_start_y =  (40,40)

        self.display_face(frame, squares, grid_start_x, grid_start_y)

        if self.colors_locked:
            frame, squares = self.resolveSquares(frame, self.locked_squares)
            return frame, squares

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 10, 20)
        
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=4)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        squares = []
        for contour in contours:
            if len(squares) >= 9:
                break
                
            approx = cv2.approxPolyDP(contour, .1 * cv2.arcLength(contour, True), True)
            
            if 4 <= len(approx) and cv2.contourArea(approx) >= 10000: 
                side_lengths = [np.linalg.norm(approx[i][0] - approx[(i+1) % 4][0]) for i in range(4)]
                avg_side_length = sum(side_lengths) / len(approx)
                
                if all(0.8 * avg_side_length <= side <= 1.2 * avg_side_length for side in side_lengths):
                    angles = []
                    for i in range(4):
                        v1 = approx[i][0] - approx[(i - 1) % 4][0]
                        v2 = approx[(i + 1) % 4][0] - approx[i][0]
                        angle = self.angle_between_vectors(v1, v2)
                        angles.append(angle)
                    
                    if all(80 <= angle <= 100 for angle in angles):
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            centroid = (cx, cy)
                        else:
                            centroid = (0, 0)
                        
                        median_hsv, median_bgr, median_lab = self.get_median_color(frame, contour)
                        
                        square = Square(contour, centroid, median_bgr, median_hsv, median_lab, center)
                        squares.append(square)
                        
                        x, y, w, h = cv2.boundingRect(approx)
                        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
                        cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"{median_bgr}", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if len(squares) == 9 and not self.colors_locked:
            self.colors_locked = True
            self.locked_squares = squares.copy() 
        
        cv2.putText(frame, f"Squares: {len(squares)}/9", (frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if len(squares) < 9 else (0, 255, 0), 2)
        
        frame, squares = self.resolveSquares(frame, squares)

        return frame, squares
        
    def resolveSquares(self, frame, detected_squares):
        """
        Display a sorted 3x3 grid in the top left corner of the frame, 
        filled with the median colors of detected squares.

        Args:
            frame: The input image frame
            detected_squares: List of Square objects with contour, centroid, and color information
        
        Returns:
            The frame with the 3x3 grid drawn on it, and the sorted squares
        """
        if len(detected_squares) < 9:
            return frame, detected_squares

        detected_squares.sort(key=lambda sq: sq.centroid[1])  

        rows = [detected_squares[i:i+3] for i in range(0, 9, 3)]

        for row in rows:
            row.sort(key=lambda sq: sq.centroid[0])

        sorted_squares = [sq for row in rows for sq in row]

        return frame, sorted_squares 

def main():
    camera = Camera()
    sides = 6
    allSquares = [[] for _ in range(sides)]
    order = ["yellow", "orange", "blue", "red", "green", "white"]
    instructions = ["YELLOW face with blue on top",  "ORANGE face with yellow on top", "BLUE with yellow on top", "RED with yellow on top", "GREEN with yellow on top", "WHITE with green on top"]

    for i in range(sides):
        center = order[i]
        while True:
            ret, frame = camera.camera.read()
            if not ret:
                break
                
            squares = allSquares[i]
            frame, squares = camera.detect_squares(frame, squares, i, center)
            allSquares[i] = squares
            
            cv2.putText(frame, f"Scanning Face {i+1}/6", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Show the {instructions[i]}", (250, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            if len(allSquares[i]) == 9:
                cv2.putText(frame, "Press SPACE to continue", (250, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, "Press R to reset this face", (200, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Square Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if len(allSquares[i]) == 9 and key == ord('r'):
                camera.reset()
                allSquares[i] = []

            if len(allSquares[i]) == 9 and key == ord(' '):
                camera.reset()
                break 
            
            if key == ord('q'): 
                camera.camera.release()
                cv2.destroyAllWindows()
                return

    camera.camera.release()
    cv2.destroyAllWindows()
    faces = colorsFromSquares(allSquares)
    convert = colorToLetter()
    CUBE_STRING = ""
    cubeOrder = ["yellow", "orange", "green", "white", "red", "blue"]
    for color in cubeOrder:
        for i in range(len(faces[color])):
            for j in range(len(faces[color][0])):
                CUBE_STRING += (convert[faces[color][i][j]])
    solution = sv.solve(CUBE_STRING, 19, 2)
    solution = solution.replace("3", "'")
    solution = solution.replace("1","")
    print("Faces", faces)
    print("Holding Green in front and Yellow on top")
    print(solution)




if __name__ == "__main__":
    main()