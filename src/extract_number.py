import numpy as np
import cv2
import math
import tensorflow as tf
import random as rng

# setting up the ranges for red colour
redLowerHSV = np.array([0, 170, 170])
redUpperHSV = np.array([10, 255, 255])

redLowerHSV2 = np.array([161, 170, 170])
redUpperHSV2 = np.array([179, 255, 255])

# init final image, variables, kernel
final_img = np.zeros((480, 640, 3)) + 255

kernel = np.ones((3, 3), np.uint8)

model = tf.keras.models.load_model('myEmnistModel.h5')
class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
final_str = ""

drawing = False
erasing = False

prev_cx = None
prev_cy = None

# get name of final file
win_name = input("name of image:")

# start capturing video
cap = cv2.VideoCapture(0)

# increase brightness of image
gamma = 0.5
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

while cap.isOpened():
    # input frame and increase brightness
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_bright = cv2.LUT(frame, lookUpTable)
    # threshold HSV image and dilate according to both the ranges
    im_hsv = cv2.cvtColor(frame_bright, cv2.COLOR_BGR2HSV)
    red_extract1 = cv2.inRange(im_hsv, redLowerHSV, redUpperHSV)
    red_extract1 = cv2.dilate(red_extract1, kernel, iterations=2)
    red_extract2 = cv2.inRange(im_hsv, redLowerHSV2, redUpperHSV2)
    red_extract2 = cv2.dilate(red_extract2, kernel, iterations=2)
    # add the effects from both the ranges
    red_extract = cv2.add(red_extract1, red_extract2)
    # cv2.imshow('win', red_extract)
    white_mass = np.nonzero(red_extract)
    # get mean of the threshold image
    cy = np.sum(white_mass[0]) / len(white_mass[0])
    cx = np.sum(white_mass[1]) / len(white_mass[1])
    if drawing is True and not (math.isnan(cx)) and not (math.isnan(cy)):
        cx = int(cx)
        cy = int(cy)
        if prev_cx is not None:
            if erasing is True:
                cv2.circle(final_img, (cx, cy), 10, (255, 255, 255), -1)
            else:
                cv2.line(final_img, (prev_cx, prev_cy), (cx, cy), (0, 0, 0), 4)
        og = final_img.copy()
        cv2.circle(final_img, (cx, cy), 4, (255, 0, 0), 2)
        prev_cx = cx
        prev_cy = cy
    else:
        og = final_img.copy()
        if not (math.isnan(cx)) and not (math.isnan(cy)):
            cv2.circle(final_img, (int(cx), int(cy)), 4, (255, 0, 0), 2)
        prev_cx = None
        prev_cy = None

    cv2.imshow(win_name, final_img)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break
    elif k == ord('s'):
        drawing = not drawing
    elif k == ord('e'):
        erasing = not erasing
    final_img = og.copy()

# collect alphanumeric from the final image

# find contours on final image
img_bw = final_img[:, :, 0]
thresh = cv2.inRange(img_bw, 0, 2)
(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours_poly = [None] * len(cnts)
boundRect = [None] * len(cnts)
alphanumeric_images = [None] * len(cnts)

# get bounding rectangles
boundRect = [cv2.boundingRect(c) for c in cnts]
(cnts, boundRect) = zip(*sorted(zip(cnts, boundRect), key=lambda b: b[1][0], reverse=False))
# extract images and resize them according to neural network input
for i in range(len(cnts)):
    alphanumeric_images[i] = img_bw[int(boundRect[i][1]):int(boundRect[i][1] + boundRect[i][3]),
                             int(boundRect[i][0]):int(boundRect[i][0] + boundRect[i][2])]
    alphanumeric_images[i] = cv2.resize(alphanumeric_images[i], (24, 24))
    alphanumeric_images[i] = 255 - alphanumeric_images[i]
    #    alphanumeric_images[i] = cv2.dilate(alphanumeric_images[i], kernel, iterations=1)
    alphanumeric_images[i] = cv2.copyMakeBorder(alphanumeric_images[i], 2, 2, 2, 2, cv2.BORDER_CONSTANT, None, 0)
    cv2.imshow('windah', alphanumeric_images[i])
    cv2.waitKey(0)
    alphanumeric_images[i] = alphanumeric_images[i] / 255.0
    alphanumeric_images[i] = np.reshape(alphanumeric_images[i], (1, 28, 28))
    final_str = final_str + class_mapping[np.argmax(tf.nn.softmax(model.predict(alphanumeric_images[i])))]
cap.release()
cv2.destroyAllWindows()
print(final_str)
# O->D
# 1->2,g
# 3->B
# 5->B
#
