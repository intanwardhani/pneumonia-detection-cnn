import os 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 
from .transforms import create_data_transforms 
from config import Config

config = Config()
class PneumoniaDataset(Dataset): 
    """Custom dataset for redistributed pneumonia X-ray images with 3-class structure"""
    
    def __init__(self, data_dir: str, split_name: str, transform=None, class_names=None):
        """
        Args:
            data_dir: Base directory (e.g., 'data/redistributed')
            split_name: 'train', 'val', or 'test'
            transform: Image transforms
            class_names: ['NORMAL', 'BACTERIA', 'VIRAL']
        """
        
        self.data_dir = data_dir
        self.split_name = split_name
        self.transform = transform
        self.class_names = class_names or config.CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        self._load_data()

    def _load_data(self):
        """Load all image paths and labels from new 3-class structure"""
        
        split_dir = os.path.join(self.data_dir, self.split_name)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Load images from each class directory
        for class_name in self.class_names:
            class_dir = os.path.join(split_dir, class_name)
            
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(image_path)
                        self.labels.append(self.class_to_idx[class_name])
                        
                        # Optional: Log class distribution
                        if len(self.image_paths) == 1:  # First image of each class
                            print(f"Found {class_name} images in {self.split_name}")
        
        print(f"Loaded {len(self.image_paths)} images from {self.split_name} split")
        
        # Verify class distribution
        class_counts = {cls: 0 for cls in self.class_names}
        for label in self.labels:
            class_name = self.class_names[label]
            class_counts[class_name] += 1
        
        print(f"Class distribution in {self.split_name}:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, 
                       image_size=config.IMAGE_SIZE, medical_transforms=True, 
                       num_workers=config.NUM_WORKERS, class_names=None): 
    """Create train, validation, and test dataloaders for redistributed data"""
    
    # Set default class names if not provided
    if class_names is None:
        class_names = config.CLASS_NAMES

    # Create transforms
    transforms_dict = create_data_transforms(image_size, medical_transforms)

    # Create datasets for each split
    train_dataset = PneumoniaDataset(
        data_dir=data_dir,
        split_name='train',
        transform=transforms_dict['train'],
        class_names=class_names
    )

    val_dataset = PneumoniaDataset(
        data_dir=data_dir,
        split_name='val',
        transform=transforms_dict['val'],
        class_names=class_names
    )

    test_dataset = PneumoniaDataset(
        data_dir=data_dir,
        split_name='test',
        transform=transforms_dict['test'],
        class_names=class_names
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Keep shuffling for training!
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No shuffling for test
        num_workers=num_workers
    )

    print(f"\nDataLoaders created for {data_dir}:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader

def create_redistributed_dataloaders(data_dir=config.DATA_REDIST_DIR,
                                     batch_size=config.BATCH_SIZE, 
                                     image_size=config.IMAGE_SIZE, 
                                     medical_transforms=True, 
                                     num_workers=config.NUM_WORKERS): 
    """Convenience function specifically for redistributed data"""
    
    return create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        medical_transforms=medical_transforms,
        num_workers=num_workers,
        class_names=config.CLASS_NAMES
        )
    
def verify_dataset_loading(data_dir=config.DATA_REDIST_DIR): 
    """Verify that datasets are loaded correctly"""
    
    print("=== Verifying Dataset Loading ===")
    
    try:
        # Create a simple transform for verification
        from torchvision import transforms
        simple_transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor()
        ])
        
        # Test each split
        for split in config.SPLIT_NAMES:
            dataset = PneumoniaDataset(
                data_dir=data_dir,
                split_name=split,
                transform=simple_transform
            )
            
            print(f"\n{split.upper()} Split:")
            print(f"  Total samples: {len(dataset)}")
            
            # Check class distribution
            class_counts = {0: 0, 1: 0, 2: 0}  # NORMAL, BACTERIA, VIRAL
            for _, label in dataset:
                class_counts[label] += 1
            
            class_names = config.CLASS_NAMES
            for i, count in class_counts.items():
                print(f"  {class_names[i]}: {count} samples")
                
            # Test a sample batch
            if len(dataset) > 0:
                sample_image, sample_label = dataset[0]
                print(f"  Sample image shape: {sample_image.shape}") # type: ignore
                print(f"  Sample label: {sample_label} ({class_names[sample_label]})")
        
        print("Dataset loading verification complete!")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        raise
    
def analyse_dataloader_classes(dataloader, class_names, num_batches=5):
    """Analyse class distribution in dataloader (handles both tensor and int labels)"""
    
    class_distribution = {name: 0 for name in class_names}
    total_samples = 0
    
    print("Analysing dataloader class distribution...")
    print("-" * 50)
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        print(f"Batch {batch_idx + 1}:")
        print(f"  Images: {len(images)}")
        
        # Handle different label formats
        if hasattr(labels, 'numpy'):  # It's a tensor
            labels_list = labels.cpu().numpy().tolist()
        else:  # It's already a list/tuple
            labels_list = list(labels)
            
        print(f"  Labels: {labels_list}")
        
        for label in labels_list:
            # Convert to integer if it's still a tensor
            if hasattr(label, 'item'):  # It's a tensor
                label_int = label.item()
            else:
                label_int = int(label)
            
            # Get class name
            class_name = class_names[label_int]
            class_distribution[class_name] += 1
            total_samples += 1
    
    print(f"Overall distribution (first {num_batches} batches):")
    for class_name, count in class_distribution.items():
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
def debug_dataloader_labels(dataloader, num_batches=1):
    """Debug function to understand label format"""
    
    print("Debugging dataloader label format...")
    print("-" * 40)
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels type: {type(labels)}")
        print(f"  Labels: {labels}")
        
        if hasattr(labels, 'shape'):
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels dtype: {labels.dtype}")
        
        # Try different conversion methods
        try:
            labels_numpy = labels.numpy()
            print(f"  Labels as numpy: {labels_numpy}")
            print(f"  Labels as numpy list: {labels_numpy.tolist()}")
        except Exception as e:
            print(f"  Error converting to numpy: {e}")
        
        # Check individual labels
        for i, label in enumerate(labels):
            print(f"    Label {i}: {type(label)} = {label}")
            if hasattr(label, 'item'):
                print(f"      Label {i} as int: {label.item()}")
            else:
                print(f"      Label {i} as int: {int(label)}")
        
        break  # Only debug first batch




    
    


