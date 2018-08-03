import numpy as np
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    
    data = img.reshape((-1,3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 7, 0.5)
    K = 2
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # seperate colors
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)
    labels_reshape = label.reshape(img.shape[0], img.shape[1])
    
    # mask
    img_mask = np.copy(img)
    # the label change everytime 
    img_mask[labels_reshape==0] = (0,0,0)
    
    cv2.imshow('orginal', img)
    cv2.imshow('kmeans', res2)
    cv2.imshow('mask', img_mask)
    
    if cv2.waitKey(5) & 0xff == 7:
        break

cv2.destroyAllWindows()