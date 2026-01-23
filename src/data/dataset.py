from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from transform import create_data_transforms

class PneumoniaDataset(Dataset):
    """Custom dataset for pneumonia X-ray images"""
    
    def __init__(self, data_dir, transform=None, class_names=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = class_names or ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """Load all image paths and labels from directory"""
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloaders(data_dir, batch_size=32, image_size=(224, 224), 
                      medical_transforms=True, num_workers=4):
    """Create train, validation, and test dataloaders"""
    
    # Create transforms
    transforms_dict = create_data_transforms(image_size, medical_transforms)
    
    # Create datasets
    train_dataset = PneumoniaDataset(
        os.path.join(data_dir, 'train'), 
        transform=transforms_dict['train']
    )
    val_dataset = PneumoniaDataset(
        os.path.join(data_dir, 'val'), 
        transform=transforms_dict['val']
    )
    test_dataset = PneumoniaDataset(
        os.path.join(data_dir, 'test'), 
        transform=transforms_dict['test']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
