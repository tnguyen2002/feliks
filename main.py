

from camera import capture_images
from vision import process_color_images
from utils import colorToLetter
import twophase.solver  as sv

def main():
    capture_images()
    faces = process_color_images("images/scramble_3/")
    order = ["yellow", "orange", "green", "white", "red", "blue"]
    convert = colorToLetter()
    CUBE_STRING = ""
    for color in order:
        for i in range(len(faces[color])):
            for j in range(len(faces[color][0])):
                CUBE_STRING += (convert[faces[color][i][j]])
    solution = sv.solve(CUBE_STRING, 19, 2)
    solution = solution.replace("3", "'")
    solution = solution.replace("1","")
    print("Holding Green in front and Yellow on top")
    print(solution)


if __name__ == "__main__":
    main()