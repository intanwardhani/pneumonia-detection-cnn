import os
import yaml
import torch

class Config:
    """Configuration class for pneumonia detection project"""
    
    # Project paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "chest_xray")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    
    # Model configuration
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 2  # NORMAL and PNEUMONIA
    FREEZE_BASE = True
    PRETRAINED = True
    
    # Data configuration
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    USE_MEDICAL_TRANSFORMS = True
    CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
    
    # Training configuration
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WEIGHT_DECAY = 1e-4
    
    # Loss function configuration
    CLASS_WEIGHTS = None  # Will be calculated if None
    
    # Evaluation metrics
    METRICS = ["accuracy", "precision", "recall", "f1_score"]
    
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
