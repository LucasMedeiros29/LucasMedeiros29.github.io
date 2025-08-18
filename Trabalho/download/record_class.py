#!/usr/bin/env python3

'''
Este script é utilizado para tirar as fotos dos objetos em múltiplas posições, a fim de otimizar a gama de variações para um melhor treinamento do modelo.
A biblioteca utilizada é o OpenCV (pip install opencv-python)
Desenvolvedores do projeto:
Gustavo Henrique Germano Ledandeck 11201810786
Caio Vilor Brandão 11201921101
Lucas Pereira De Medeiros  
'''
import cv2, pathlib, sys, os

CLASSES = ["low", "medium", "high", "normal"]
CLASS = sys.argv[1].lower()
assert CLASS in CLASSES, f"Choose one of {CLASSES}"

SAVE_DIR = pathlib.Path("data/raw") / CLASS
SAVE_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
print(f"🔴  Recording for class: {CLASS}")
print("SPACE  – capture frame")
print("Q      – quit")

count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        fname = SAVE_DIR / f"{CLASS}_{count:03d}__.jpg"
        cv2.imwrite(str(fname), frame)
        print(f"Saved {fname}")
        count += 1
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
