import numpy as np
import cv2
#79 200 30
redLower = np.array([0, 0, 140])
redUpper = np.array([40, 40, 255])

kernel = np.ones((5, 5), np.uint8)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    red_extract = cv2.inRange(frame, redLower, redUpper)
    red_extract = cv2.erode(red_extract, kernel, iterations=2)
    red_extract = cv2.morphologyEx(red_extract, cv2.MORPH_OPEN, kernel)
    red_extract = cv2.dilate(red_extract, kernel, iterations=1)
    #cv2.imshow('win', red_extract)
    # Find contours in the image
    (cnts, _) = cv2.findContours(red_extract.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(frame, cnts, -1, (0, 255, 0), 3)
    cv2.imshow('win2', img)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
