import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18
from config import Config

config = Config()
def create_transfer_learning_model(model_name: str='resnet18', 
                                   num_classes: int=config.NUM_CLASSES, 
                                   freeze_base: bool=config.FREEZE_BASE) -> nn.Module:
    """Create transfer learning model with config support"""
    
    # Load base model
    if model_name == 'resnet18':
        base_model = models.resnet18(weights=config.TRAIN_WEIGHTS)
        num_features = base_model.fc.in_features
    elif model_name == 'resnet50':
        base_model = models.resnet50(weights=config.TRAIN_WEIGHTS)
        num_features = base_model.fc.in_features
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Create new classifier
    new_classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)  # NO softmax/sigmoid here!
    )
    
    # Replace the original fc layer
    base_model.fc = new_classifier  # type: ignore
    
    # Freeze base model if requested
    if freeze_base:
        for param in base_model.parameters():
            param.requires_grad = False
        # Unfreeze new layers
        for param in base_model.fc.parameters():
            param.requires_grad = True
    
    return base_model

def create_resnet18_eval(num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    Create a ResNet18 model with custom output layer for transfer learning.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights (default: False)
    
    Returns:
        torch.nn.Module: ResNet18 model with custom classifier
    """
    
    # Load pretrained ResNet18
    model = resnet18(pretrained=pretrained)
    
    # Create the exact same architecture as training
    num_features = model.fc.in_features
    model.fc = nn.Sequential(           # type: ignore
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    return model


