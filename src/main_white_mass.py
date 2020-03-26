import numpy as np
import cv2
import math

redLowerHSV = np.array([0, 170, 170])
redUpperHSV = np.array([10, 255, 255])

redLowerHSV2 = np.array([161, 170, 170])
redUpperHSV2 = np.array([179, 255, 255])

final_img = np.zeros((480, 640, 3))+255

kernel = np.ones((5, 5), np.uint8)

drawing = False
erasing = False

prev_cx = None
prev_cy = None

win_name = input("name of image:")

cap = cv2.VideoCapture(0)

gamma = 0.5
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_bright = cv2.LUT(frame, lookUpTable)
    #print(frame.shape)
    im_hsv = cv2.cvtColor(frame_bright, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', im_hsv)
    red_extract1 = cv2.inRange(im_hsv, redLowerHSV, redUpperHSV)
    red_extract1 = cv2.dilate(red_extract1, kernel, iterations=2)
    red_extract2 = cv2.inRange(im_hsv, redLowerHSV2, redUpperHSV2)
    red_extract2 = cv2.dilate(red_extract2, kernel, iterations=2)
    red_extract = cv2.add(red_extract1, red_extract2)
    cv2.imshow('win', red_extract)
    white_mass = np.nonzero(red_extract)
    cy = np.sum(white_mass[0])/len(white_mass[0])
    cx = np.sum(white_mass[1])/len(white_mass[1])
    if drawing is True and not(math.isnan(cx)) and not(math.isnan(cy)):
        cx = int(cx)
        cy = int(cy)
        if prev_cx is not None:
            if erasing is True:
                cv2.circle(final_img, (cx,cy), 10, (255, 255, 255), -1)
            else:
                cv2.line(final_img, (prev_cx, prev_cy), (cx, cy), (0, 0, 0), 2)
        og = final_img.copy()
        cv2.circle(final_img, (cx, cy), 4, (255, 0, 0), 2)
        prev_cx = cx
        prev_cy = cy
    else:
        og = final_img.copy()
        if not(math.isnan(cx)) and not(math.isnan(cy)):
            cv2.circle(final_img, (int(cx), int(cy)), 4, (255, 0, 0), 2)
        prev_cx = None
        prev_cy = None
    #cv2.imshow('winrand',frame)
    cv2.imshow('winbright', frame_bright)
    cv2.imshow(win_name, final_img)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break
    elif k == ord('s'):
        drawing = not drawing
    elif k == ord('e'):
        erasing = not erasing
    final_img = og.copy()
win_name = win_name + ".jpg"
#cv2.imwrite(win_name, final_img)
cap.release()
cv2.destroyAllWindows()
