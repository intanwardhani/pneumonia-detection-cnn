# utils.py

import os 
import matplotlib.pyplot as plt 
import numpy as np 
import torch 
from torchvision.utils import make_grid 
from config import Config

config = Config()
def analyse_dataset(data_dir):
    """Analyse dataset distribution and statistics for 3 classes""" 
    
    class_names = config.CLASS_NAMES 
    class_counts = {cls: 0 for cls in class_names}
    
    # Handle NORMAL class
    normal_dir = os.path.join(data_dir, 'NORMAL')
    if os.path.exists(normal_dir):
        class_counts['NORMAL'] = len([
            f for f in os.listdir(normal_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    # Handle PNEUMONIA class (split into BACTERIA and VIRUS)
    pneumonia_dir = os.path.join(data_dir, 'PNEUMONIA')
    if os.path.exists(pneumonia_dir):
        for img_name in os.listdir(pneumonia_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename_lower = img_name.lower()
                if 'bacteria' in filename_lower:
                    class_counts['BACTERIA'] += 1
                elif 'virus' in filename_lower:
                    class_counts['VIRUS'] += 1
                else:
                    # Default to BACTERIA if subtype is not specified
                    class_counts['BACTERIA'] += 1

    total_images = sum(class_counts.values())
    print(f"Dataset Analysis:")
    for cls, count in class_counts.items():
        percentage = (count / total_images) * 100
        print(f"  {cls}: {count} images ({percentage:.1f}%)")

    # Plot distribution with different colors for each class
    colors = ['blue', 'red', 'green']  # NORMAL, BACTERIA, VIRUS
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=colors)
    plt.title('Dataset Class Distribution (3 Classes)')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')

    plt.show()

    return class_counts, total_images

def show_batch(dataloader, num_images=config.NUM_BATCH_DISPLAY, title="Sample Batch"): 
    """Show a batch of images from the dataloader""" 
    
    # Get a batch of images and labels 
    images, labels = next(iter(dataloader))
    
    # Denormalise images
    if images.shape[1] == 3:  # RGB images
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.229, 0.224])
    else:  # Grayscale converted to 3-channel
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])

    images = images * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    images = torch.clamp(images, 0, 1)

    # Create grid
    grid = make_grid(images[:num_images], nrow=4, padding=2)

    # Convert to numpy and plot
    np_grid = grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(12, 8))
    plt.imshow(np_grid)
    plt.title(title)
    plt.axis('off')
    plt.show()

def get_class_weights(data_dir): 
    """Calculate class weights for handling imbalance in 3-class classification""" 
    
    class_names = config.CLASS_NAMES 
    class_counts = {cls: 0 for cls in class_names}
    
    # Handle NORMAL class
    normal_dir = os.path.join(data_dir, 'NORMAL')
    if os.path.exists(normal_dir):
        class_counts['NORMAL'] = len([
            f for f in os.listdir(normal_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    # Handle PNEUMONIA class (split into BACTERIA and VIRUS)
    pneumonia_dir = os.path.join(data_dir, 'PNEUMONIA')
    if os.path.exists(pneumonia_dir):
        for img_name in os.listdir(pneumonia_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename_lower = img_name.lower()
                if 'bacteria' in filename_lower:
                    class_counts['BACTERIA'] += 1
                elif 'virus' in filename_lower:
                    class_counts['VIRUS'] += 1
                else:
                    # Default to BACTERIA if subtype is not specified
                    class_counts['BACTERIA'] += 1

    total = sum(class_counts.values())
    num_classes = len(class_names)
    weights = {}

    print("Class distribution and weights:")
    for cls, count in class_counts.items():
        weight = total / (num_classes * count) if count > 0 else 1.0
        weights[cls] = weight
        print(f"{cls}: {count} images, weight: {weight:.3f}")

    return weights

def get_class_weights_tensor(data_dir): 
    """Get class weights as a PyTorch tensor for loss function""" 
    
    weights_dict = get_class_weights(data_dir) 
    class_names = config.CLASS_NAMES

    # Convert dictionary to tensor in the correct order
    weight_values = [weights_dict[cls] for cls in class_names]
    
    return torch.tensor(weight_values)

def get_filename_pneumonia_type(filename): 
    """Helper function to determine pneumonia type from filename"""
    
    filename_lower = filename.lower()
    
    if 'bacteria' in filename_lower: 
        return 'BACTERIA' 
    elif 'virus' in filename_lower:
        return 'VIRUS' 
    else: 
        return 'UNKNOWN'

