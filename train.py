import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from src.data.dataset import create_dataloaders
from src.models.transfer_learning import create_transfer_learning_model
from src.training.trainer import train_model

def main():
    # Load configuration (can override with custom YAML)
    config = Config()
    
    # Or load from YAML file
    # config = Config.load_from_yaml('config/resnet18.yaml')
    
    print(f"Using device: {config.DEVICE}")
    
    # Create data loaders using config
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        image_size=tuple(config.IMAGE_SIZE),
        medical_transforms=config.USE_MEDICAL_TRANSFORMS,
        num_workers=config.NUM_WORKERS
    )
    
    # Create model using config
    model = create_transfer_learning_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        freeze_base=config.FREEZE_BASE
    )
    model = model.to(config.DEVICE)
    
    # Train model using config
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE
    )
    
    return model, history

if __name__ == "__main__":
    main()
