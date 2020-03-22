import cv2
import numpy as np

img = cv2.imread('stuff.jpg')
cv2.imshow('image', img)
cv2.waitKey(0)
lighter = img[312:464, 318:363]
img[312:464, 370:(370 + 363 - 318)] = lighter
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
