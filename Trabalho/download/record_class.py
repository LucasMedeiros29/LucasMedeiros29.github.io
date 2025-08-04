#Lucas Pereira de Medeiros
#Gustavo Henrique Germano Ledandeck
#Caio Vilor Brandão
#04/08/2025
#Programa: record
#chamada: python3 record_class.py
import cv2, pathlib, sys, os

CLASSES = ["low", "medium", "high", "normal"]
CLASS = sys.argv[1].lower()
assert CLASS in CLASSES, f"Choose one of {CLASSES}"

SAVE_DIR = pathlib.Path("data/raw") / CLASS
SAVE_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
print(f"Gravando para classe: {CLASS}")
print("Espaço - capturar frame")
print("Q      – sair")

count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        fname = SAVE_DIR / f"{CLASS}_{count:03d}.jpg"
        cv2.imwrite(str(fname), frame)
        print(f"Salvo {fname}")
        count += 1
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
