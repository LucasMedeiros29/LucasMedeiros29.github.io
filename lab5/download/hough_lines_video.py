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
out_left = cv2.VideoWriter('output_left.mp4', fourcc, fps, (frame_width, frame_height))
out_right = cv2.VideoWriter('output_right.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if not ret1 or not ret2:
        print("Falha na captura.")
        break

    # Pré-processamento
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Ajuste fino no Canny para reduzir ruído
    edges_left = cv2.Canny(gray_left, 150, 250)
    edges_right = cv2.Canny(gray_right, 150, 250)

    # Ajuste fino no HoughLinesP
    lines_left = cv2.HoughLinesP(edges_left, 1, np.pi / 180, threshold=100, minLineLength=25, maxLineGap=20)
    lines_right = cv2.HoughLinesP(edges_right, 1, np.pi / 180, threshold=100, minLineLength=25, maxLineGap=20)

    # Desenhar linhas
    if lines_left is not None:
        for line in lines_left:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if lines_right is not None:
        for line in lines_right:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_right, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Mostrar e gravar
    cv2.imshow("Left - Lines", frame_left)
    cv2.imshow("Right - Lines", frame_right)

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
