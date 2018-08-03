import numpy as np
import cv2

def fourier_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = gray/255
    
    trans = np.fft.fft2(norm)
    f_shift = np.fft.fftshift(trans)
    ft_result = 20*np.log(np.abs(f_shift))
    
    return ft_result

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    ft = fourier_transform(img)
    
    cv2.imshow('original', img)
    cv2.imshow('ft', ft)
    
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
cap.release()       
cv2.destroyAllWindows()