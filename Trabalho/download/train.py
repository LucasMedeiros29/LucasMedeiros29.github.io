#Lucas Pereira de Medeiros
#Gustavo Henrique Germano Ledandeck
#Caio Vilor Brandão
#04/08/2025
#Programa: train
#chamada: python3 train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import lr_scheduler
import cv2
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


data_dir = "data/processed"  # --->>> aqui dataset ja tratado pelo prepare

num_classes = 4

num_epochs = 100 

learning_rate = 0.001

batch_size = 32


mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   #----> params para tratar a imagem media e desvio padrao


class AlbumentationsDataset(datasets.ImageFolder):  #---> func tirada do stackoverflow que ajuda na criacao de dir p/ o model
    def __init__(self, root, transform=None):
        super(AlbumentationsDataset, self).__init__(root)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        image = cv2.imread(path)
        if image is None:
            print(f"Não foi possível ler imagem {path}.")
            return torch.randn(3, 224, 224), target
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, target


tf_train = A.Compose([
    A.Resize(height=224, width=224),
    
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.8),
 
    A.Perspective(scale=(0.05, 0.1), p=0.7),     

    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.GaussianBlur(p=0.2),    
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])
tf_val = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

train_dataset = AlbumentationsDataset(root=os.path.join(data_dir, 'train'), transform=tf_train)
val_dataset = AlbumentationsDataset(root=os.path.join(data_dir, 'val'), transform=tf_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Achado {len(train_dataset)} imagens no set de treinamento.")
print(f"Achado {len(val_dataset)} imagens no set de validação.")
print(f"Classes: {train_dataset.classes}")


model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


print("\n--- Iniciando ---")
best_acc = 0.0

os.makedirs("models", exist_ok=True)
with open('models/class_names.pkl', 'wb') as f:
    pickle.dump(train_dataset.classes, f)
print("Nomes das classes salvas em modelos/nome_classes.pkl")


for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  
            loader = train_loader
        else:
            model.eval()   
                           
            loader = val_loader

        running_loss = 0.0
        running_corrects = 0


        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):  #--> garante que ta na etapa de treino
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1) 
                loss = criterion(outputs, labels) 
        
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step() 

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects.double() / len(loader.dataset)

        print(f'{phase.capitalize()} Perda: {epoch_loss:.4f} Precisão: {epoch_acc:.4f}')

        
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "models/best_resnet18.pth")
            print("Modelo salvo")

print("\n--- Finalizado ---")
print(f"Melhor precisão: {best_acc:.4f}")

