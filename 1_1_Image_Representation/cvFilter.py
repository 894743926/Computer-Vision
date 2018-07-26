import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

cap = cv2.VideoCapture(0)

while True:
    # get camera data
    _, frame = cap.read()
    # convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    lower_blue = np.array([30, 82.5, 102.5])
    upper_blue = np.array([40, 92.5, 195])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    res = cv2.bitwise_and(frame, frame, mask= mask)

    # plt.imshow(frame, cmap='hsv')
    # plt.imshow(mask, cmap='gray')
    # plt.imshow(res, cmap='hsv')
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
