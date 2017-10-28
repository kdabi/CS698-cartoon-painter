import cv2, sys
import numpy as np
import os
'''
 * Main program begins here.
'''
# read command-line filename argument
path = sys.argv[1]
path_train = './train'
path_test = './test'
path_val = './val'
i=0
valI =0
valTrain =0
valTest =0
for filename in os.listdir(path):
    # load original image as grayscale
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith("._jpeg_"):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        newfilename = filename.replace('._jpeg_', '.jpeg')

        # set up display window with trackbars for minimum and maximum threshold
        # values
        i = i+1
        if i < 401:
            output_path = path_train
            valTrain = valTrain +1
            #newfilename = filename
            newfilename = str(valTrain)+'.jpg'
        elif i < 501:
            valI = valI +1
            newfilename = str(valI)+'.jpg'
            output_path = path_val
        else:
            valTest = valTest +1
            newfilename = str(valTest)+'.jpg'
            output_path = path_test
        image_original = cv2.imread(os.path.join(path, filename))
        minT = 50
        maxT = 90
        edge = cv2.Canny(img, minT, maxT)
        inverted_edge = cv2.bitwise_not(edge)
        sketch = cv2.cvtColor(inverted_edge, cv2.COLOR_GRAY2BGR)
        final_image = np.concatenate((image_original, sketch), axis=1)
        cv2.imwrite(os.path.join(output_path, newfilename), final_image)

