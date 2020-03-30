import numpy as np
import cv2

final_img = np.zeros((24, 24)) + 255
cv2.imshow('lol', final_img)
dst = cv2.copyMakeBorder(final_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, None, 0)
cv2.imshow('bordered', dst)
cv2.waitKey(0)
print(dst.shape)
cv2.destroyAllWindows()
