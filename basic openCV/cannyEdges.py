import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg', 0)
lb = 100
ub = 200
cv2.imshow('win2',img)
while True:
    edges = cv2.Canny(img, lb, ub)
    cv2.imshow('win', edges)
    k = cv2.waitKey(0) & 0xff
    if k == ord('w'):
        ub += 1
    elif k == ord('s'):
        ub -= 1
    elif k == ord('a'):
        lb -= 1
    elif k == ord('d'):
        lb += 1
    elif k == ord('q'):
        break
    print(lb)
    print(ub)
    print("\n\n")

cv2.destroyAllWindows()
