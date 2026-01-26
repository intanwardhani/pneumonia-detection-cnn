import os 
import shutil 
import random  
import argparse
from collections import defaultdict
from config import Config

config = Config()
class DataRedistributor: 
    """Redistribute pneumonia dataset to fix class imbalance and distribution issues"""
    
    def __init__(self, original_dir=config.DATA_DIR, output_dir=config.DATA_REDIST_DIR):
        self.original_dir = original_dir
        self.output_dir = output_dir
        self.class_mapping = {
            'NORMAL': 'NORMAL',
            'bacteria': 'BACTERIA',
            'virus': 'VIRAL'
        }
        
    def count_original_data(self):
        """Count images in original dataset"""
        
        counts = defaultdict(lambda: defaultdict(int))
        
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.original_dir, split)
            if not os.path.exists(split_dir):
                continue
                
            # Count NORMAL
            normal_dir = os.path.join(split_dir, 'NORMAL')
            if os.path.exists(normal_dir):
                counts['NORMAL'][split] = len([
                    f for f in os.listdir(normal_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
            
            # Count PNEUMONIA subtypes
            pneumonia_dir = os.path.join(split_dir, 'PNEUMONIA')
            if os.path.exists(pneumonia_dir):
                for img_name in os.listdir(pneumonia_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if 'bacteria' in img_name.lower():
                            counts['BACTERIA'][split] += 1
                        elif 'virus' in img_name.lower():
                            counts['VIRAL'][split] += 1
                        else:
                            counts['BACTERIA'][split] += 1  # Default to bacteria
        
        return counts

    def collect_all_images(self):
        """Collect all images with their paths and original splits"""
        
        all_images = defaultdict(list)
        
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.original_dir, split)
            if not os.path.exists(split_dir):
                continue
                
            # Collect NORMAL images
            normal_dir = os.path.join(split_dir, 'NORMAL')
            if os.path.exists(normal_dir):
                for img_name in os.listdir(normal_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(normal_dir, img_name)
                        all_images['NORMAL'].append((img_path, split))
            
            # Collect PNEUMONIA images
            pneumonia_dir = os.path.join(split_dir, 'PNEUMONIA')
            if os.path.exists(pneumonia_dir):
                for img_name in os.listdir(pneumonia_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(pneumonia_dir, img_name)
                        if 'bacteria' in img_name.lower():
                            subtype = 'BACTERIA'
                        elif 'virus' in img_name.lower():
                            subtype = 'VIRAL'
                        else:
                            subtype = 'BACTERIA'
                        all_images[subtype].append((img_path, split))
        
        return all_images

    def create_output_structure(self):
        """Create output directory structure"""
        
        os.makedirs(self.output_dir, exist_ok=True)
        for split in config.SPLIT_NAMES:   
            for cls in config.CLASS_NAMES:
                os.makedirs(os.path.join(self.output_dir, split, cls), exist_ok=True)

    def redistribute_data_balanced(self, model_type='resnet18'):
        """Redistribute data for equal class distribution per split"""
        
        print("=== Original Data Distribution ===")
        original_counts = self.count_original_data()
        
        for cls, split_counts in original_counts.items():
            print(f"\n{cls}:")
            for split, count in split_counts.items():
                print(f"  {split}: {count}")
        
        # Collect all images
        all_images = self.collect_all_images()
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Shuffle images within each class
        for cls in all_images:
            random.shuffle(all_images[cls])
        
        # Define target distribution based on model type
        if model_type == 'resnet18':
            train_targets = {'NORMAL': 900, 'BACTERIA': 900, 'VIRAL': 450}
            val_targets = {'NORMAL': 200, 'BACTERIA': 200, 'VIRAL': 100}
            test_targets = {'NORMAL': 200, 'BACTERIA': 200, 'VIRAL': 100}
        else:  # resnet50
            train_targets = {'NORMAL': 1100, 'BACTERIA': 1100, 'VIRAL': 550}
            val_targets = {'NORMAL': 200, 'BACTERIA': 200, 'VIRAL': 100}
            test_targets = {'NORMAL': 200, 'BACTERIA': 200, 'VIRAL': 100}
        
        print(f"\n=== Target Distribution for {model_type.upper()} ===")
        for split_name, targets in [('train', train_targets), ('val', val_targets), ('test', test_targets)]:
            print(f"\n{split_name.upper()}:")
            for cls, count in targets.items():
                print(f"  {cls}: {count} images")
        
        # Select target number of images from each class
        selected_images = {}
        for cls in config.CLASS_NAMES:
            available = len(all_images[cls])
            target = train_targets[cls] + val_targets[cls] + test_targets[cls]
            
            if available >= target:
                selected_images[cls] = all_images[cls][:target]
                print(f"Selected {target}/{available} {cls} images")
            else:
                selected_images[cls] = all_images[cls]
                print(f"Warning: Only {available} {cls} images available, requested {target}")
        
        # Redistribute into train, val, test splits
        train_images = {}
        val_images = {}
        test_images = {}
        
        for cls in config.CLASS_NAMES:
            class_images = selected_images[cls]
            
            # Take validation samples first (ensure we get enough)
            val_images[cls] = class_images[:val_targets[cls]]
            
            # Take test samples
            test_images[cls] = class_images[val_targets[cls]:val_targets[cls] + test_targets[cls]]
            
            # Remaining go to training
            train_images[cls] = class_images[val_targets[cls] + test_targets[cls]:]
        
        # Ensure we meet minimum targets
        for cls in config.CLASS_NAMES:
            if len(train_images[cls]) < train_targets[cls]:
                # Take from remaining images if available
                remaining = selected_images[cls][len(train_images[cls]) + len(val_images[cls]) + len(test_images[cls]):]
                needed = train_targets[cls] - len(train_images[cls])
                train_images[cls].extend(remaining[:needed])
        
        # Print final distribution
        print("\n=== Final Redistributed Distribution ===")
        for split_name, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            print(f"\n{split_name.upper()}:")
            total_split = 0
            for cls, img_list in images.items():
                print(f"  {cls}: {len(img_list)} images")
                total_split += len(img_list)
            print(f"  Total: {total_split} images")
        
        # Create output structure
        self.create_output_structure()
        
        # Copy images to new structure
        for split_name, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            for cls, img_list in images.items():
                for img_path, original_split in img_list:
                    img_name = os.path.basename(img_path)
                    new_path = os.path.join(self.output_dir, split_name, cls, img_name)
                    shutil.copy2(img_path, new_path)
        
        print(f"\n✅ Redistributed data created in: {self.output_dir}")
        print(f"   - Original data preserved in: {self.original_dir}")
        print(f"   - Optimised for: {model_type.upper()}")
        
        return {
            'train': {cls: len(img_list) for cls, img_list in train_images.items()},
            'val': {cls: len(img_list) for cls, img_list in val_images.items()},
            'test': {cls: len(img_list) for cls, img_list in test_images.items()}
        }
        
def main(): 
    parser = argparse.ArgumentParser(
        description='Redistribute pneumonia dataset for balanced training'
        ) 
    parser.add_argument(
        '--original_dir', default=config.DATA_DIR, 
        help='Original dataset directory'
        ) 
    parser.add_argument(
        '--output_dir', default=config.DATA_REDIST_DIR, 
        help='Output directory for redistributed data'
        ) 
    parser.add_argument(
        '--model_type', choices=['resnet18', 'resnet50'], 
        default='resnet18', 
        help='Target model architecture'
        )
    
    args = parser.parse_args()
    redistributor = DataRedistributor(args.original_dir, args.output_dir)
    new_distribution = redistributor.redistribute_data_balanced(args.model_type)

    print("\n=== Final Summary ===")
    for split, counts in new_distribution.items():
        print(f"{split}: {counts}")
        
if __name__ == "__main__": 
    main()



    
    