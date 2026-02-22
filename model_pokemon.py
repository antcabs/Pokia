import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

class CardGraderModel(nn.Module):
    def __init__(self):
        super(CardGraderModel, self).__init__()
        
        # Utiliser un modèle pré-entraîné (ResNet18)
        self.backbone = models.resnet18(pretrained=True)
        
        # Remplacer la dernière couche
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Couches pour les prédictions
        self.predictor = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)  # 5 sorties: coins, bords, surface, centrage, note_psa
        )
    
    def forward(self, x):
        features = self.backbone(x)
        predictions = self.predictor(features)
        return predictions

# Transformations pour préparer les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path):
    """Charge le modèle entraîné"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CardGraderModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def prepare_image(image):
    """Prépare l'image pour le modèle"""
    # Si l'image est un array numpy (format OpenCV)
    if isinstance(image, np.ndarray):
        # Convertir BGR à RGB si nécessaire
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
    # Sinon, si c'est un chemin de fichier
    elif isinstance(image, str):
        image_pil = Image.open(image).convert('RGB')
    # Si c'est déjà un objet PIL
    else:
        image_pil = image
    
    # Appliquer les transformations et ajouter dimension de batch
    image_tensor = transform(image_pil).unsqueeze(0)
    return image_tensor

def analyze_card(model, device, image):
    """Analyse une carte avec le modèle ML"""
    # Préparer l'image
    image_tensor = prepare_image(image).to(device)
    
    # Faire la prédiction
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Extraire les prédictions
    coins = float(predictions[0, 0].item())
    bords = float(predictions[0, 1].item())
    surface = float(predictions[0, 2].item())
    centrage = float(predictions[0, 3].item())
    grade_psa = float(predictions[0, 4].item())
    
    # Normaliser les scores dans leurs plages respectives
    coins = min(30, max(0, coins * 30))  # 0-30
    bords = min(30, max(0, bords * 30))  # 0-30
    surface = min(20, max(0, surface * 20))  # 0-20
    centrage = min(20, max(0, centrage * 20))  # 0-20
    grade_psa = min(10, max(1, grade_psa * 9 + 1))  # 1-10
    
    # Calculer la note globale
    note_globale = coins + bords + surface + centrage
    
    return {
        'coins': coins,
        'bords': bords,
        'surface': surface,
        'centrage': centrage,
        'note_globale': note_globale,
        'grade_psa': int(grade_psa)
    }