import cv2, sys
import os
'''
 * Main program begins here.
'''
# read command-line filename argument
input_directory = sys.argv[1]
output_directory = sys.argv[2]

for filename in os.listdir(input_directory):
    # load original image as grayscale
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith("._jpeg_"):
        img = cv2.imread(input_directory + "/" +filename, cv2.IMREAD_GRAYSCALE)
        newfilename = filename.replace('._jpeg_', '.jpeg')

        # set up display window with trackbars for minimum and maximum threshold
        # values
        minT = 50
        maxT = 90
        edge = cv2.Canny(img, minT, maxT)
        inverted_edge = cv2.bitwise_not(edge)
        cv2.imwrite(output_directory + "/" + newfilename, inverted_edge)

