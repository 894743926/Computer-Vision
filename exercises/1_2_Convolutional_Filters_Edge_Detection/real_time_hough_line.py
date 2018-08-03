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

def draw_hough_line(img):
    
    canny_thresh = (75, 150)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_canny = cv2.Canny(img_gray, canny_thresh[0], canny_thresh[1])
    
    rho = 1  # resulotion
    theta = np.pi/180
    threshold = 100
    minLineLength = 60
    maxLineGap = 20
    
    lines = cv2.HoughLinesP(img_canny, rho, theta, threshold, minLineLength, maxLineGap)
    
    img_copy = np.copy(img)
    
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                img_copy = cv2.line(img_copy, (x1, y1), (x2, y2), (240, 240, 0), 3)
    except:
        print('Find no line')
    
    return img_canny, img_copy

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    
    img_canny, img_hough = draw_hough_line(img)
    
    cv2.imshow('original', img)
    cv2.imshow('canny', img_canny)
    cv2.imshow('hough_line', img_hough)
    
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
cap.release()       
cv2.destroyAllWindows()