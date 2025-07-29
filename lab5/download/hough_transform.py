import cv2
import numpy as np
import os


#OBS A EXECUCAO DO SCRIPT TA NO MAIN LA NO FINAL!!!!

# Função para aplicar a Transformada de Hough em imagens
def hough_transform(image_path):
    # Ler a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return

    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar Canny para detecção de bordas
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Aplicar a Transformada de Hough para linhas
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    # Desenhar as linhas detectadas
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Aplicar a Transformada de Hough para círculos
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

    # Desenhar os círculos detectados
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Desenhar o círculo externo
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
            # Desenhar o centro do círculo
            cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)

    # Mostrar os resultados
    cv2.imshow("Detected Lines and Circles", img)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
	
	hough_transform("frame_caio_left%d.jpg"%0)
