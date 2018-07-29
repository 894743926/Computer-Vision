import numpy as np
import cv2

def fourier_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = gray/255
    
    trans = np.fft.fft2(norm)
    f_shift = np.fft.fftshift(trans)
    ft_result = 20*np.log(np.abs(f_shift))
    
    return ft_result

def filter_img(img):
    kernal = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    blur = cv2.GaussianBlur(img, (5,5), 0)
    return cv2.filter2D(blur, -1, kernal.T+kernal)

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ft = cv2.Canny(img, 50, 150)
    
    cv2.imshow('original', img)
    cv2.imshow('filter', 255-ft)
    
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
cap.release()       
cv2.destroyAllWindows()