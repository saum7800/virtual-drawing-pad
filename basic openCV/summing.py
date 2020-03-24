import numpy as np
import cv2

img = cv2.imread('stuff.jpg')
img = cv2.inRange(img, )
print(img)
print(np.sum(img, axis=2)/3)