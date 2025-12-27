import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

def train_classifier(data_dir: str, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
    print(f"Starting training with data from {data_dir}...")
    
    # 1. Setup Data Augmentation & Loading
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Check if 'train' and 'val' folders exist, otherwise assume flat structure and split?
    # For simplicity, we assume standard ImageFolder structure: data/image_class/image.jpg
    # And we'll just use it all for training if 'train' subdir doesn't exist.
    
    if os.path.exists(os.path.join(data_dir, 'train')):
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    else:
        print("No train/val split found. Using all data for training.")
        full_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
        image_datasets = {'train': full_dataset}
        
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in image_datasets
    }
    
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")

    # 2. Setup Model (MobileNetV2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace head
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)

    # 3. Training Loop
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase not in dataloaders: continue
            
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 4. Save Model
    os.makedirs('artifacts', exist_ok=True)
    save_path = 'artifacts/place_classifier.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 5. Save Class Names
    import json
    class_path = 'artifacts/classes.json'
    with open(class_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {class_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/images', help='Path to image data')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    
    train_classifier(args.data_dir, args.epochs)
