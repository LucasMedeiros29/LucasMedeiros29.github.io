import cv2
import numpy as np

# IDs das câmeras
cam_left = 4
cam_right = 6

# Inicializar captura
cap_left = cv2.VideoCapture(cam_left)
cap_right = cv2.VideoCapture(cam_right)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Erro ao abrir uma das câmeras.")
    exit()

# Configuração do gravador de vídeo
frame_width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_left = cv2.VideoWriter('output_left_circles.mp4', fourcc, fps, (frame_width, frame_height))
out_right = cv2.VideoWriter('output_right_circles.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if not ret1 or not ret2:
        print("Falha na captura.")
        break

    # Pré-processamento
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Blur para reduzir ruído
    blur_left = cv2.medianBlur(gray_left, 5)
    blur_right = cv2.medianBlur(gray_right, 5)

    # Detectar círculos com HoughCircles
    circles_left = cv2.HoughCircles(
        blur_left, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=100, param2=30, minRadius=10, maxRadius=60
    )
    circles_right = cv2.HoughCircles(
        blur_right, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=100, param2=30, minRadius=10, maxRadius=60
    )

    # Desenhar círculos
    if circles_left is not None:
        circles_left = np.uint16(np.around(circles_left))
        for i in circles_left[0, :]:
            cv2.circle(frame_left, (i[0], i[1]), i[2], (0, 255, 0), 2)  # contorno
            cv2.circle(frame_left, (i[0], i[1]), 2, (0, 0, 255), 3)      # centro

    if circles_right is not None:
        circles_right = np.uint16(np.around(circles_right))
        for i in circles_right[0, :]:
            cv2.circle(frame_right, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame_right, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Mostrar e gravar
    cv2.imshow("Left - Circles", frame_left)
    cv2.imshow("Right - Circles", frame_right)

    out_left.write(frame_left)
    out_right.write(frame_right)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

# Liberar recursos
cap_left.release()
cap_right.release()
out_left.release()
out_right.release()
cv2.destroyAllWindows()
