import numpy as np
import cv2


redLower = np.array([0, 0, 120])
redUpper = np.array([90, 90, 255])

redLowerHSV = np.array([0, 170, 170])
redUpperHSV = np.array([10, 255, 255])

final_img = np.zeros((480, 640, 3))+255

kernel = np.ones((5, 5), np.uint8)

drawing = False
erasing = False

prev_cx = None
prev_cy = None

win_name = input("name of image:")

cap = cv2.VideoCapture(0)

#increasing brightness of image
gamma = 0.5
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
#    print(frame.shape)
    frame_bright = cv2.LUT(frame, lookUpTable)
    im_hsv = cv2.cvtColor(frame_bright, cv2.COLOR_BGR2HSV)
    red_extract = cv2.inRange(im_hsv, redLowerHSV, redUpperHSV)
    red_extract = cv2.dilate(red_extract, kernel, iterations=1)
    #cv2.imshow('win', red_extract)
    # Find contours in the image
    (cnts, _) = cv2.findContours(red_extract.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0 and drawing is True:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        M = cv2.moments(cnts[0])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if prev_cx is not None:
            if abs(cx - prev_cx) <= 30 and abs(cy - prev_cy) <= 30:
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
        prev_cx = None
        prev_cy = None

    img = cv2.drawContours(frame, cnts, -1, (0, 255, 0), 3)
    cv2.imshow('win', frame_bright)
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
cv2.imwrite(win_name, final_img)
cap.release()
cv2.destroyAllWindows()
