import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from config import Config

config = Config()
def analyse_dataset(data_dir):
    """Analyse dataset distribution and statistics"""
    class_names = config.CLASS_NAMES
    class_counts = {cls: 0 for cls in class_names}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            class_counts[class_name] = len([
                f for f in os.listdir(class_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
    
    total_images = sum(class_counts.values())
    print(f"Dataset Analysis:")
    for cls, count in class_counts.items():
        percentage = (count / total_images) * 100
        print(f"  {cls}: {count} images ({percentage:.1f}%)")
    
    # Plot distribution
    plt.figure(figsize=(8, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color=['blue', 'red'])
    plt.title('Dataset Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()
    
    return class_counts, total_images

def show_batch(dataloader, num_images=16, title="Sample Batch"):
    """Show a batch of images from the dataloader"""
    # Get a batch of images and labels
    images, labels = next(iter(dataloader))
    
    # Denormalise images
    if images.shape == 3:  # RGB images
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
    """Calculate class weights for handling imbalance"""
    class_names = config.CLASS_NAMES
    class_counts = {cls: 0 for cls in class_names}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            class_counts[class_name] = len([
                f for f in os.listdir(class_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
    
    total = sum(class_counts.values())
    weights = {}
    for cls, count in class_counts.items():
        weights[cls] = total / (len(class_names) * count)
    
    return weights
