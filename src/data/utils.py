import os
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import make_grid 
from typing import Dict, Tuple, Optional
from .dataset import PneumoniaDataset
from config import Config

config = Config()
def show_batch(dataloader: DataLoader, num_images: int = 8, 
               title: str = "Sample Batch", class_names: list=config.CLASS_NAMES):
    """Show batch with class information"""
    
    try:
        # Get a batch of images and labels 
        images, labels = next(iter(dataloader))
        
        # Denormalise images
        if hasattr(config, 'MEAN') and hasattr(config, 'STD'):
            mean = torch.tensor(config.MEAN)
            std = torch.tensor(config.STD)
        else:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.229, 0.224])

        images = images * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
        images = torch.clamp(images, 0, 1)

        # Create grid
        grid = make_grid(images[:num_images], nrow=4, padding=2)
        np_grid = grid.numpy().transpose((1, 2, 0))
        
        # Create figure with title and info
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Display image grid
        ax1.imshow(np_grid)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Display class information
        class_counts = {}
        for label in labels[:num_images]:
            if class_names is not None and isinstance(label, int):
                class_name = class_names[label]
            else:
                class_name = f"Class {label}"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Create class info text
        info_text = "Classes in batch: "
        for class_name, count in class_counts.items():
            info_text += f"{class_name}({count}) "
        
        ax2.text(0.5, 0.5, info_text, ha='center', va='center', 
                fontsize=12, transform=ax2.transAxes)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed information
        print(f"\n{title}")
        print(f"Total images: {len(images)}")
        print("Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
        
    except Exception as e:
        print(f"Error in show_batch_with_info: {e}")


def get_class_weights(data_dir: str, use_training_only: bool =True) -> Tuple[Dict[str, float], Dict[str, int]]: 
    """Calculate class weights for handling imbalance in 3-class classification.
    Args:
        data_dir (str): Path to the data directory containing redistributed folders
        use_training_only (bool): If True, only use training split for weights

    Returns:
        dict: Dictionary mapping class names to their weights
        dict: Dictionary mapping class names to their counts
    """ 
    
    class_names = config.CLASS_NAMES 
    class_counts = {cls: 0 for cls in class_names}
    
    # Determine which splits to use
    if use_training_only:
        splits_to_use = ['train']
    else:
        splits_to_use = config.SPLIT_NAMES
    
    for split_name in splits_to_use:
        split_dir = os.path.join(data_dir, split_name)
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory not found: {split_dir}")
            continue
        
        print(f"\nProcessing {split_name} split:")

        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            try:
                # Count images in each class directory
                image_files = [
                    f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ]
                count = len(image_files)
                class_counts[class_name] += count
                
                print(f"  {class_name}: {count} images")
                
            except Exception as e:
                print(f"Error processing {class_dir}: {e}")
                continue

    total = sum(class_counts.values())
    num_classes = len(class_names)

    # Calculate class weights using inverse frequency method
    weights = {}
    print("\nClass distribution and weights:")
    print("-" * 50)

    for cls, count in class_counts.items():
        if count == 0:
            # Handle zero-count classes by giving them the highest weight
            weight = 10.0  # High penalty for missing classes
            print(f"{cls}: {count} images, weight: {weight:.3f} (ZERO COUNT - HIGH PENALTY)")
        else:
            # Standard inverse frequency weighting
            weight = total / (num_classes * count)
            weights[cls] = weight
            print(f"{cls}: {count} images, weight: {weight:.3f}")

    print("-" * 50)
    print(f"Total images: {total}")

    return weights, class_counts

def get_balanced_sampling_weights(data_dir: str, use_training_only: bool = True) -> Dict[str, float]: 
    """Get sampling weights for balanced DataLoader sampling.
    This returns weights that can be used with WeightedRandomSampler
    to ensure balanced batches during training.

    Args:
        data_dir (str): Path to the data directory
        use_training_only (bool): If True, only use training split for weights

    Returns:
        dict: Dictionary mapping class names to sampling weights
    """
    
    _, class_counts = get_class_weights(data_dir, use_training_only)

    # Calculate sampling weights (inverse of class frequencies)
    total_samples = sum(class_counts.values())
    sampling_weights = {}

    for cls, count in class_counts.items():
        if count == 0:
            sampling_weights[cls] = 1.0  # Default weight for missing classes
        else:
            sampling_weights[cls] = total_samples / (len(class_counts) * count)

    return sampling_weights

def get_class_weights_tensor(data_dir: str, 
                             device: Optional[torch.device]) -> torch.Tensor:
    """Get class weights as a PyTorch tensor for loss function
    
    Args:
        data_dir (str): Path to the data directory
        device (torch.device, optional): Device to place the tensor on
    
    Returns:
        torch.Tensor: Tensor of class weights in the order specified by config.CLASS_NAMES
    """
    
    # Get class weights dictionary
    weights_dict, _ = get_class_weights(data_dir)
    
    # Get class names from config
    config = Config()
    class_names = config.CLASS_NAMES
    
    # Convert dictionary to tensor in the correct order
    weight_values = [weights_dict[cls] for cls in class_names]
    
    # Create tensor and move to device if specified
    weights_tensor = torch.tensor(weight_values, dtype=torch.float32)
    
    if device is not None:
        weights_tensor = weights_tensor.to(device)
    
    return weights_tensor

def get_data_statistics(data_dir: str) -> Dict[str, Dict[str, int]]: 
    """Get comprehensive statistics about the dataset.
    Args:
    data_dir (str): Path to the data directory

    Returns:
        dict: Statistics organized by split and class
    """
    
    class_names = config.CLASS_NAMES
    split_names = config.SPLIT_NAMES

    stats = {}

    for split_name in split_names:
        stats[split_name] = {}
        split_dir = os.path.join(data_dir, split_name)
        
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory not found: {split_dir}")
            continue
            
        total_in_split = 0
        
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            
            if os.path.exists(class_dir):
                image_files = [
                    f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ]
                count = len(image_files)
                stats[split_name][class_name] = count
                total_in_split += count
            else:
                stats[split_name][class_name] = 0
        
        stats[split_name]['TOTAL'] = total_in_split

    return stats

def print_dataset_statistics(data_dir: str): 
    """Print formatted dataset statistics""" 
    
    stats = get_data_statistics(data_dir)
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()} SET:")
        print("-" * 30)
        
        for class_name, count in split_stats.items():
            if class_name != 'TOTAL':
                print(f"  {class_name:10}: {count:5} images")
        
        print(f"  {'TOTAL':10}: {split_stats['TOTAL']:5} images")

    print("\n" + "="*60)
    
def create_weighted_sampler(data_dir: str, split: str = 'train', use_training_only: bool = True): 
    """Create a WeightedRandomSampler for balanced training.
    Args:
    data_dir (str): Path to data directory
    split (str): Which split to use
    use_training_only (bool): Whether to use only training split

    Returns:
        torch.utils.data.WeightedRandomSampler: Sampler for balanced sampling
    """
    
    sampling_weights = get_balanced_sampling_weights(data_dir, use_training_only)

    # Create list of weights for each sample
    config = Config()
    dataset = PneumoniaDataset(data_dir=data_dir, split_name=split)

    # Get class indices for each sample
    class_to_idx = {cls: idx for idx, cls in enumerate(config.CLASS_NAMES)}
    sample_weights = [sampling_weights[config.CLASS_NAMES[class_to_idx[dataset[i][1]]]] 
                    for i in range(len(dataset))]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
        )

if __name__ == "__main__":
    data_dir = config.DATA_REDIST_DIR
    print("Testing data utility functions...")

    # Test class weights
    print("\n1. Testing get_class_weights...")
    weights, counts = get_class_weights(data_dir)
    print(f"Class weights: {weights}")
    print(f"Class Counts: {counts}")

    # Test class weights tensor
    print("\n2. Testing get_class_weights_tensor...")
    weights_tensor = get_class_weights_tensor(data_dir, device=torch.device('cpu'))
    print(f"Class weights tensor: {weights_tensor}")

    # Test dataset statistics
    print("\n3. Testing dataset statistics...")
    print_dataset_statistics(data_dir)

