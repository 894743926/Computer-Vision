import cv2
from models import Net
import torch

cap = cv2.VideoCapture(0)

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
net = Net('saved_models/keypoints_model_well.pt')
net.eval()

while True:
    _, image = cap.read()
    
    faces = face_cascade.detectMultiScale(image, 1.2, 2)
    
    # make a copy of the original image to plot detections on
    image_with_detections = image.copy()
    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)
        
        # Select the region of interest that is the face in the image 
        roi = image_with_detections[y:y+h, x:x+w]
    
        ## TODO: Convert the face region from RGB to grayscale
        gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
        norm_img = gray_img/255
    
        ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
        fit_size_img = cv2.resize(norm_img, (100,100))
    
        ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        torch_img = torch.from_numpy(fit_size_img).unsqueeze(0).unsqueeze(1).type(torch.FloatTensor)
    
        ## TODO: Make facial keypoint predictions using your loaded, trained network 
        ## perform a forward pass to get the predicted facial keypoints
        predicted_key_pts = net(torch_img)
        # reshape to batch_size x 68 x 2 pts
        predicted_key_pts = predicted_key_pts.view(predicted_key_pts.size()[0], 68, -1)
        predicted_key_pts = predicted_key_pts.data.numpy()
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*50.0+100
        predicted_key_pts = predicted_key_pts[0]
        # coordinate
        predicted_key_pts[:,0] = predicted_key_pts[:, 0]/100*norm_img.shape[1] + x
        predicted_key_pts[:,1] = predicted_key_pts[:, 1]/100*norm_img.shape[0] + y
    
        for (xx, yy) in predicted_key_pts:
            cv2.circle(image_with_detections, (xx, yy), 2, (230, 0, 230), 3)
        
     
    cv2.imshow('origin', image)
    cv2.imshow('with faces', image_with_detections)
    
    if cv2.waitKey(4) & 0xff == 27:
          break
cv2.destroyAllWindows()    