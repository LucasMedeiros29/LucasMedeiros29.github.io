import numpy as np
import cv2 as cv


sift = cv.SIFT_create()


cap1 = cv.VideoCapture(0)  # Webcam esquerda
cap2 = cv.VideoCapture(2)  # Webcam direita

while True:
    
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()

    if not ret1 or not ret2:
        print("Erro ao capturar vídeo das webcams.")
        break

    
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > 10:  
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = gray1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    else:
        print("Não foram encontrados matches suficientes - {}/{}".format(len(good), 10))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # desenhar matches em verde
                       singlePointColor=None,
                       matchesMask=matchesMask,  
                       flags=2)

    img3 = cv.drawMatches(gray1, kp1, gray2, kp2, good, None, **draw_params)

    # Mostre o resultado
    cv.imshow('Matches', img3)

    # Pressione 'q' para sair do loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Libere as webcams e feche as janelas
cap1.release()
cap2.release()
cv.destroyAllWindows()
