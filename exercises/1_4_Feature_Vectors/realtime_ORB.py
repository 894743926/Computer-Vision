import cv2
import copy
import numpy as np

train_img = cv2.imread('images/my_thump.jpg')
train_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(2000, 2.0)
bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING,
                   crossCheck = True)

cascade = cv2.CascadeClassifier("../1_2_Convolutional_Filters_Edge_Detection/detector_architectures/haarcascade_frontalface_default.xml")


keypoints_train, descriptors_train = orb.detectAndCompute(train_gray, None)


cap = cv2.VideoCapture(0)
while True:
    _, query_img = cap.read()
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    
    # find the faces and mask the other things
    result_faces = cascade.detectMultiScale(query_gray)
    draw_img = np.zeros_like(query_gray)
    for x, y, w, h in result_faces:
        draw_img[y:y+h, x:x+w] = [255]
    
    # find keypoints and match with the train img
    keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, draw_img)
    matchs = bf.match(descriptors_train, descriptors_query)
    
    matchs = sorted(matchs, key=lambda x: x.distance)
    result_img = copy.copy(train_img)
    
    result_img = cv2.drawMatches(train_img, keypoints_train, query_img, keypoints_query, matchs[:50], result_img, flags=2)
    
    cv2.imshow('match result' ,result_img)
    
    if cv2.waitKey(6) & 0xff == 27:
        break
        
cv2.destroyAllWindows()   

