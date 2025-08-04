#Lucas Pereira de Medeiros
#Gustavo Henrique Germano Ledandeck
#Caio Vilor Brandão
#04/08/2025
#Programa: prepare dataset
#chamada: python3 prepare_dataset.py
import pathlib
import shutil
import random
import cv2
import numpy as np
import os
from tqdm import tqdm


RAW_DATA_DIR = pathlib.Path("data/raw")

PROCESSED_DATA_DIR = pathlib.Path("data/processed")

TRAIN_VAL_SPLIT = 0.8  

CLASSES = ["low", "medium", "high", "normal"]

PADDING = 0.10 

def crop_object_from_image(img_path):
    frame = cv2.imread(str(img_path))
    if frame is None:
        return None


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 30, 150)


    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None 


    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)

 
    pad_w = int(w * PADDING)
    pad_h = int(h * PADDING)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame.shape[1], x + w + pad_w)
    y2 = min(frame.shape[0], y + h + pad_h)


    cropped_img = frame[y1:y2, x1:x2]
    return cropped_img

def prepare_dataset():

    if PROCESSED_DATA_DIR.exists():
        print(f"Removendo diretório de dados processados já existente: {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)

    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    train_dir = PROCESSED_DATA_DIR / "train"
    val_dir = PROCESSED_DATA_DIR / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    print(f"Preparando dataset em: {PROCESSED_DATA_DIR}\n")

    for cls in CLASSES:
        print(f"Processando classe: {cls}")
        
        (train_dir / cls).mkdir()
        (val_dir / cls).mkdir()
        
        raw_class_dir = RAW_DATA_DIR / cls
        if not raw_class_dir.exists():
            print(f"Diretório para classes '{cls}' não encontrado.")
            continue
            
        image_files = list(raw_class_dir.glob("*.jpg"))
        random.shuffle(image_files)
        
        split_idx = int(len(image_files) * TRAIN_VAL_SPLIT)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        print(f"  - {len(train_files)} imagens para treinamento")
        print(f"  - {len(val_files)} imagens para validação")
        
        for f in tqdm(train_files, desc=f"  Cropping train '{cls}'"):
            cropped = crop_object_from_image(f)
            if cropped is not None:
                cv2.imwrite(str(train_dir / cls / f.name), cropped)
            
        for f in tqdm(val_files, desc=f"  Cropping val '{cls}'"):
            cropped = crop_object_from_image(f)
            if cropped is not None:
                cv2.imwrite(str(val_dir / cls / f.name), cropped)
            
    print("\n Preparação do dataset completa")

if __name__ == "__main__":
    prepare_dataset()
