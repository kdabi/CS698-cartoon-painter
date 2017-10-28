import os,sys
import numpy as np
import cv2
path = sys.argv[1]
files = os.listdir(path)
i=1

for File in  files:
    image = cv2.imread(os.path.join(path, File))
    os.remove(os.path.join(path, File))
    resized_image = cv2.resize(image, (256, 256))
    #resized_image = np.concatenate((resized_image, resized_image), axis=1)
    cv2.imwrite(os.path.join(path,str(i)+'.jpg'), resized_image)
    i=i+1

