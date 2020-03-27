import numpy as np
import cv2

img = cv2.imread('messi5.jpg')

img_resized = cv2.resize(img, (1500, 1000))

cv2.imshow('og', img)
cv2.imshow('resized', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img.shape)
print(img_resized.shape)