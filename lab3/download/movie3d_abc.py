import numpy as np
import cv2

# IDs das câmeras (ajuste conforme necessário: geralmente 0 e 1)
CamL_id = 0
CamR_id = 1

# Captura ao vivo das duas webcams
CamL = cv2.VideoCapture(CamL_id)
CamR = cv2.VideoCapture(CamR_id)

# Verifica se as câmeras abriram corretamente
if not CamL.isOpened() or not CamR.isOpened():
    print("Erro ao abrir as webcams.")
    exit()

# Leitura dos parâmetros de calibração estéreo
print("Lendo parâmetros de calibração...")
cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

# Loop de captura ao vivo
while True:
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()
    
    if retL and retR:
        # Conversão para cinza (caso deseje usar posteriormente)
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Remapeamento com as retificações
        Left_nice = cv2.remap(imgL, Left_Stereo_Map_x, Left_Stereo_Map_y,
                              cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        Right_nice = cv2.remap(imgR, Right_Stereo_Map_x, Right_Stereo_Map_y,
                               cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # Criação da imagem anáglifa (vermelho + ciano)
        output = cv2.merge((
            Right_nice[:, :, 0],  # Canal azul
            Right_nice[:, :, 1],  # Canal verde
            Left_nice[:, :, 2]    # Canal vermelho
        ))

        output = cv2.resize(output, (700, 700))
        cv2.imshow("Câmera Estéreo 3D Ao Vivo", output)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Erro na leitura das câmeras.")
        break

# Libera os recursos
CamL.release()
CamR.release()
cv2.destroyAllWindows()