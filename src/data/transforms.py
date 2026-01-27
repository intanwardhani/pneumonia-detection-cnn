# import torch  # for clahe if used later
# import random
# from PIL import Image   # for histogram equalization if used later
import torchvision.transforms as transforms
from config import Config

config = Config()
class MedicalImageTransforms: 
    """Medical image transforms optimised for X-ray pneumonia detection with 3 classes"""
    
    @staticmethod
    def get_train_transforms(image_size=config.IMAGE_SIZE, medical_mode=True):
        """Transforms for training with data augmentation"""
        if medical_mode:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(image_size),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.229, 0.224])
            ])

    @staticmethod
    def get_val_transforms(image_size=config.IMAGE_SIZE, medical_mode=True):
        """Transforms for validation (no augmentation)"""
        
        if medical_mode:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.229, 0.224])
            ])

    @staticmethod
    def get_test_transforms(image_size=config.IMAGE_SIZE, medical_mode=True):
        """Transforms for testing (same as validation)"""
        return MedicalImageTransforms.get_val_transforms(image_size, medical_mode)

    @staticmethod
    def get_balanced_transforms_3class(image_size=config.IMAGE_SIZE):
        """
        Create balanced transforms for 3-class classification
        Different augmentation strategies for each class to handle imbalance
        """
        
        # Standard transforms for validation/test
        standard_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
        
        # Enhanced transforms for NORMAL class (minority class)
        normal_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            # transforms.RandomHorizontalFlip(p=0.8),      # Higher probability
            # transforms.RandomRotation(degrees=20),        # More rotation
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # More translation
            transforms.ColorJitter(brightness=0.3, contrast=0.3),      # More variation
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
        
        # Standard transforms for BACTERIA class
        bacteria_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            # transforms.RandomHorizontalFlip(p=0.5),      # Standard augmentation
            # transforms.RandomRotation(degrees=15),        # Moderate rotation
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
        
        # Standard transforms for VIRUS class
        virus_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            # transforms.RandomHorizontalFlip(p=0.5),      # Standard augmentation
            # transforms.RandomRotation(degrees=15),        # Moderate rotation
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
        
        return normal_transform, bacteria_transform, virus_transform, standard_transform

    @staticmethod
    def get_custom_augmentation(transform_type='clahe'):
        """Get specific augmentation techniques"""
        if transform_type == 'clahe':
            # Implement CLAHE here if needed
            pass
        elif transform_type == 'histogram_equalization':
            # Histogram equalization for better contrast
            pass
        return None


def create_data_transforms(image_size=config.IMAGE_SIZE, medical_transforms=True): 
    """Create all transforms for train, val, and test""" 
    return { 
            'train': MedicalImageTransforms.get_train_transforms(image_size, medical_transforms), 
            'val': MedicalImageTransforms.get_val_transforms(image_size, medical_transforms), 
            'test': MedicalImageTransforms.get_test_transforms(image_size, medical_transforms)
            }
