import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

def create_transfer_learning_model(model_name='resnet18', num_classes=2, freeze_base=True):
    """Create transfer learning model with config support"""
    config = Config()
    
    # Override with parameters if provided
    if num_classes != config.NUM_CLASSES:
        config.NUM_CLASSES = num_classes
    if model_name != config.MODEL_NAME:
        config.MODEL_NAME = model_name
    
    # Load base model
    if model_name == 'resnet18':
        base_model = models.resnet18(pretrained=config.PRETRAINED)
        num_features = base_model.fc.in_features
    elif model_name == 'resnet50':
        base_model = models.resnet50(pretrained=config.PRETRAINED)
        num_features = base_model.fc.in_features
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Modify output layer
    base_model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, config.NUM_CLASSES),
        nn.Softmax(dim=1) if config.NUM_CLASSES > 1 else nn.Sigmoid()
    )
    
    # Freeze base model if requested
    if freeze_base:
        for param in base_model.parameters():
            param.requires_grad = False
        # Unfreeze new layers
        for param in base_model.fc.parameters():
            param.requires_grad = True
    
    return base_model
