import numpy as np
import cv2
#79 200 30
redLower = np.array([0, 0, 120])
redUpper = np.array([90, 90, 255])
redLowerHSV = np.array([0, 150, 150])
redUpperHSV = np.array([30, 255, 255])
final_img = np.zeros((480, 640, 3))+255
kernel = np.ones((5, 5), np.uint8)
prev_cx = None
prev_cy = None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    print(frame.shape)
    im_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_extract = cv2.inRange(im_hsv, redLowerHSV, redUpperHSV)
#    red_extract = cv2.morphologyEx(red_extract, cv2.MORPH_OPEN, kernel)
    red_extract = cv2.dilate(red_extract, kernel, iterations=1)
    cv2.imshow('win', red_extract)
    # Find contours in the image
    (cnts, _) = cv2.findContours(red_extract.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        M = cv2.moments(cnts[0])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if prev_cx is not None:
            cv2.line(final_img, (prev_cx, prev_cy), (cx, cy), (0, 0, 0), 2)
        prev_cx = cx
        prev_cy = cy

        #cv2.circle(final_img, (cx, cy), 10, (0, 0, 255), -1)

    img = cv2.drawContours(frame, cnts, -1, (0, 255, 0), 3)
    cv2.imshow('win2', img)
    cv2.imshow('drawWin', final_img)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
