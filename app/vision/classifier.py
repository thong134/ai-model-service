import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from typing import List, Dict

class PlaceClassifier:
    def __init__(self, model_path: str = None, num_classes: int = 6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load classes dynamically
        import json
        classes_path = 'artifacts/classes.json'
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                self.classes = json.load(f)
        else:
            # Fallback (User should retrain to match their folders)
            self.classes = ['beach', 'mountain', 'temple', 'urban', 'forest', 'historical']
            
        self.num_classes = len(self.classes)
        
        # Load MobileNetV2
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze base layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace classifier head
        self.model.classifier[1] = nn.Linear(self.model.last_channel, self.num_classes)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path: str) -> Dict[str, float]:
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
            top_prob, top_idx = torch.max(probabilities, 0)
            predicted_class = self.classes[top_idx.item()]
            
            return {
                "class": predicted_class,
                "confidence": float(top_prob.item()),
                "all_scores": {
                    cls: float(prob) for cls, prob in zip(self.classes, probabilities)
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def train(self, data_dir: str, epochs: int = 5):
        # Implementation of training loop would go here
        # Using ImageFolder dataset
        pass
