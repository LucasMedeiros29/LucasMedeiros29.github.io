import numpy as np
import cv2
import time


# Check for left and right camera IDs
CamL_id = 2
CamR_id = 0
'''
CamL= cv2.VideoCapture(CamL_id)
CamR= cv2.VideoCapture(CamR_id)

for i in range(100):
    retL, frameL= CamL.read()
    retR, frameR= CamR.read()

cv2.imshow('imgL',frameL)
cv2.imshow('imgR',frameR)
'''
'''
if cv2.waitKey(0) & 0xFF == ord('y') or cv2.waitKey(0) & 0xFF == ord('Y'):
    CamL_id = 1
    CamR_id = 2
    print("Camera IDs maintained")

elif cv2.waitKey(0) & 0xFF == ord('n') or cv2.waitKey(0) & 0xFF == ord('N'):
    CamL_id = 2
    CamR_id = 1
    print("Camera IDs swapped")
else:
    print("Wrong input response")
    exit(-1)
CamR.release()
CamL.release()
'''

CamL= cv2.VideoCapture(CamL_id)
CamR= cv2.VideoCapture(CamR_id)


start = time.time()
T = 10
count = 0
i = 0
while True:
    timer = T - int(time.time() - start)
    retR, frameR= CamR.read()
    retL, frameL= CamL.read()
    
    img1_temp = frameL.copy()
    cv2.putText(img1_temp,"%r"%timer,(50,50),1,5,(55,0,0),5)
    

    
    
    cv2.imshow('imgR',frameR)
    cv2.imshow('imgL',frameL)

    # Find the chess board corners
    #retR, cornersR = cv2.findChessboardCorners(grayR,(8,6),None)
    #retL, cornersL = cv2.findChessboardCorners(grayL,(8,6),None)

    # If corners are detected in left and right image then we save it.
    # Save on "s" key or exit on "q"
    k = cv2.waitKey(1) 
    if  k == ord('s'):
        cv2.imwrite('Caio_right_%d.png'%i,frameR)
        cv2.imwrite('Caio_left_%d.png'%i,frameL)
    
        i = i + 1
        print("frame", i)
		
    elif k == ord('q'):
        break
    
    # Press esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("Closing the cameras!")
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
