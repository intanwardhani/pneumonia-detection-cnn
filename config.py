import os
import yaml
import torch

class Config:
    """Configuration class for pneumonia detection project"""
    
    # Project paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "chest_xray")
    DATA_REDIST_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "redistributed")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    
    # Model configuration
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 3  # NORMAL, BACTERIA, VIRAL
    FREEZE_BASE = True
    PRETRAINED = True
    TRAIN_WEIGHTS = "ResNet18_Weights.DEFAULT"
    
    # Data configuration
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_BATCH_DISPLAY = 16
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.25, 0.25, 0.25]  # for x-ray images, less aggressive normalisation

    USE_MEDICAL_TRANSFORMS = True
    CLASS_NAMES = ['NORMAL', 'BACTERIA', 'VIRAL']
    SPLIT_NAMES = ['train', 'val', 'test']
    
    # Training configuration
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WEIGHT_DECAY = 1e-4
    
    # Loss function configuration
    CLASS_WEIGHTS = None  # Will be calculated if None
    
    # Evaluation metrics
    METRICS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    
    @classmethod
    def load_from_yaml(cls, yaml_path):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config instance and update with YAML values
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config

# Global instance
config = Config()
