import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg', 0)
lb = 100
ub = 200
cv2.imshow('win2', img)
ret, thresh = cv2.threshold(img, 127, 255, 0)
cv2.imshow('win', thresh)
edges = cv2.Canny(img, 100, 200)
cv2.imshow('win3', edges)
edges2 = cv2.Canny(thresh, 100, 200)
cv2.imshow('win', edges2)
cv2.waitKey(0)
cv2.destroyAllWindows()