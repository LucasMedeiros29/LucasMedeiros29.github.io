import cv2
import numpy as np
import sys
import glob

def onTrackbarChange(max_slider):
    global img
    global dst

    dst = np.copy(img)

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=max_slider, param2=30, minRadius=0, maxRadius=0)

    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Circulo de fora
            cv2.circle(dst, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Centro do Ciruclo
            cv2.circle(dst, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Result Image", dst)

if __name__ == "__main__":
    
    images = glob.glob("frame_caio_left%d.jpg"%0)  

    for image_path in images:
        img = cv2.imread(image_path)
        
        
        dst = np.copy(img)

        
        cv2.namedWindow("Result Image")
          
        
        initThresh = 300
        
        maxThresh = 2000

        cv2.createTrackbar("threshold", "Result Image", initThresh, maxThresh, onTrackbarChange)
        onTrackbarChange(initThresh)

        while True:
            key = cv2.waitKey(1)
            if key == 27:  # Press ESC to exit
                break

    cv2.destroyAllWindows()
