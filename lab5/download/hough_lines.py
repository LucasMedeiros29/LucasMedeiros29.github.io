import cv2
import numpy as np
import sys
import glob

def onTrackbarChange(max_slider):
    global img
    global dst
    global gray

    dst = np.copy(img)

    th1 = max_slider 
    th2 = th1 * 0.4
    edges = cv2.Canny(img, th1, th2)
    
    # Apply probabilistic hough line transform
    lines = cv2.HoughLinesP(edges, 2, np.pi/180.0, 50, minLineLength=100, maxLineGap=200)

    # Draw lines on the detected points
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imshow("Result Image", dst)    
    cv2.imshow("Edges", edges)

if __name__ == "__main__":
    # Load all images from the specified directory
    images = glob.glob("frame_caio_left%d.jpg"%0)  # Altere o caminho conforme necessário

    for image_path in images:
        img = cv2.imread(image_path)
        
        # Create a copy for later usage
        dst = np.copy(img)

        # Convert image to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create display windows
        cv2.namedWindow("Edges")
        cv2.namedWindow("Result Image")
          
        # Initialize threshold value
        initThresh = 890    #----> esse valor aqui agora ta cravado, com 890 da para contornar o chessboarder sem achar que os quadradinhos objetos tbm, e contorna a luz no teto, o monitor atras e etc.
        # Maximum threshold value
        maxThresh = 1000

        #cv2.createTrackbar("threshold", "Result Image", initThresh, maxThresh, onTrackbarChange)  ----> essa func apenas gerava uma box com a imagem e um scroll para variar o initThresh e ver o resultado conforme oscila ele
        onTrackbarChange(initThresh)

        while True:
            key = cv2.waitKey(1)
            if key == 27:  # Press ESC to exit
                break

    cv2.destroyAllWindows()
