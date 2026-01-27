import os 
import sys
import logging 
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from typing import Dict
from config import Config 

config = Config()
class TrainingTracker: 
    """Track training metrics over epochs"""
    
    def __init__(self, save_dir: str):
        """
        Initialize training tracker
        
        Args:
            save_dir: Directory to save training logs and plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize storage
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        
        log_file = os.path.join(self.save_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def update_epoch(self, train_metrics: Dict, val_metrics: Dict):
        """
        Update metrics for current epoch
        
        Args:
            train_metrics: Training metrics for current epoch
            val_metrics: Validation metrics for current epoch
        """
        self.train_losses.append(train_metrics.get('avg_loss', 0))
        self.val_losses.append(val_metrics.get('avg_loss', 0))
        self.train_accuracies.append(train_metrics.get('accuracy', 0))
        self.val_accuracies.append(val_metrics.get('accuracy', 0))
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        
        # Log epoch results
        logging.info(f"Train Loss: {self.train_losses[-1]:.4f} | Train Acc: {self.train_accuracies[-1]:.2f}%")
        logging.info(f"Val Loss: {self.val_losses[-1]:.4f} | Val Acc: {self.val_accuracies[-1]:.2f}%")

    def plot_training_history(self, save_dir: str):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='Train Accuracy')
        axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score plot
        if self.train_metrics and 'f1' in self.train_metrics[0]:
            train_f1 = [m.get('f1', 0) for m in self.train_metrics]
            val_f1 = [m.get('f1', 0) for m in self.val_metrics]
            axes[1, 0].plot(epochs, train_f1, 'g-', label='Train F1')
            axes[1, 0].plot(epochs, val_f1, 'orange', label='Val F1')
            axes[1, 0].set_title('Training and Validation F1 Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.5, 'Training Metrics Summary', fontsize=14, fontweight='bold')
        
        # Add summary statistics
        if self.val_accuracies:
            best_epoch = np.argmax(self.val_accuracies) + 1
            best_val_acc = max(self.val_accuracies)
            
            summary_text = f"""
                Best Validation Accuracy: {best_val_acc:.2f}% 
                Best Epoch: {best_epoch} 
                Total Epochs: {len(self.train_losses)}
                Final Metrics: 
                Train Loss: {self.train_losses[-1]:.4f} 
                Val Loss: {self.val_losses[-1]:.4f} 
                Train Accuracy: {self.train_accuracies[-1]:.2f}% 
                Val Accuracy: {self.val_accuracies[-1]:.2f}% 
                """
            axes[1, 1].text(0.1, 0.3, summary_text, fontsize=10, verticalalignment='top')
    
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'training_history.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training history plot saved to {save_path}")
        plt.show()

    def save_metrics_summary(self, save_dir: str):
        """Save metrics summary to file"""
        
        save_path = os.path.join(save_dir, 'metrics_summary.txt')
        
        with open(save_path, 'w') as f:
            f.write("TRAINING METRICS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall summary
            if self.val_accuracies:
                best_epoch = np.argmax(self.val_accuracies) + 1
                best_val_acc = max(self.val_accuracies)
                
                f.write(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})\n")
                f.write(f"Final Validation Accuracy: {self.val_accuracies[-1]:.4f}\n")
                f.write(f"Total Training Epochs: {len(self.train_losses)}\n\n")
                
                # Detailed metrics per epoch
                f.write("DETAILED EPOCH METRICS:\n")
                f.write("-" * 30 + "\n")
                
                for i, (train_m, val_m) in enumerate(zip(self.train_metrics, self.val_metrics)):
                    epoch = i + 1
                    f.write(f"\nEpoch {epoch}:\n")
                    f.write(f"  Train - Loss: {train_m.get('avg_loss', 'N/A'):.4f}, "
                        f"Acc: {train_m.get('accuracy', 0):.4f}\n")
                    f.write(f"  Val   - Loss: {val_m.get('avg_loss', 'N/A'):.4f}, "
                        f"Acc: {val_m.get('accuracy', 0):.4f}\n")
            
            # Final metrics
            if self.val_metrics:
                final_val = self.val_metrics[-1]
                f.write(f"\nFINAL VALIDATION METRICS:\n")
                f.write("-" * 30 + "\n")
                
                for key, value in final_val.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
        
        logging.info(f"Metrics summary saved to {save_path}")
        
# def evaluate_model(model: nn.Module, dataloader: DataLoader, 
#                    class_names: List[str], device: torch.device, 
#                    save_dir: str = "evaluation_results") -> Dict: 
    
#     """ Comprehensive model evaluation
#     Args:
#         model: Trained model to evaluate
#         dataloader: DataLoader for evaluation
#         class_names: List of class names
#         device: Device to perform evaluation on
#         save_dir: Directory to save evaluation results

#     Returns:
#         Dictionary containing all evaluation metrics
#     """
    
#     # Create evaluation directory
#     os.makedirs(save_dir, exist_ok=True)

#     # Initialize metrics calculator
#     num_classes = len(class_names)
#     metrics = ClassificationMetrics(num_classes, class_names, device)

#     # Set model to evaluation mode
#     model.eval()

#     print("Evaluating model...")
#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
#             images, labels = images.to(device), labels.to(device)
            
#             # Forward pass
#             outputs = model(images)
            
#             # Update metrics
#             metrics.update(outputs, labels, losses)

#     # Calculate final metrics
#     final_metrics = metrics.calculate_metrics()

#     # Plot confusion matrix
#     cm_path = os.path.join(save_dir, 'confusion_matrix.png')
#     metrics.plot_confusion_matrix(save_path=cm_path)

#     # Plot ROC curve
#     roc_path = os.path.join(save_dir, 'roc_curve.png')
#     metrics.plot_roc_curve(save_path=roc_path)

#     # Print detailed results
#     print("\n" + "="*60)
#     print("MODEL EVALUATION RESULTS")
#     print("="*60)

#     print(f"Accuracy: {final_metrics['accuracy']:.4f}")

#     if num_classes == 1:
#         print(f"Precision: {final_metrics['precision']:.4f}")
#         print(f"Recall: {final_metrics['recall']:.4f}")
#         print(f"F1 Score: {final_metrics['f1']:.4f}")
#         if 'roc_auc' in final_metrics:
#             print(f"ROC AUC: {final_metrics['roc_auc']:.4f}")
#     else:
#         print(f"Macro Precision: {final_metrics.get('precision_macro', 0):.4f}")
#         print(f"Macro Recall: {final_metrics.get('recall_macro', 0):.4f}")
#         print(f"Macro F1: {final_metrics.get('f1_macro', 0):.4f}")
#         if 'roc_auc_macro' in final_metrics:
#             print(f"ROC AUC (Macro): {final_metrics['roc_auc_macro']:.4f}")

#     # Print per-class metrics
#     if 'per_class' in final_metrics:
#         print("\nPER-CLASS METRICS:")
#         print("-" * 30)
#         for class_name, class_metrics in final_metrics['per_class'].items():
#             print(f"{class_name}:")
#             print(f"  Precision: {class_metrics['precision']:.4f}")
#             print(f"  Recall: {class_metrics['recall']:.4f}")
#             print(f"  F1 Score: {class_metrics['f1']:.4f}")
#             print(f"  Support: {class_metrics['support']}")

#     # Save detailed metrics
#     import json
#     metrics_path = os.path.join(save_dir, 'detailed_metrics.json')
#     with open(metrics_path, 'w') as f:
#         # Convert numpy arrays to lists for JSON serialization
#         json_serializable = {}
#         for key, value in final_metrics.items():
#             if isinstance(value, np.ndarray):
#                 json_serializable[key] = value.tolist()
#             elif isinstance(value, dict):
#                 json_serializable[key] = {}
#                 for sub_key, sub_value in value.items():
#                     if isinstance(sub_value, (np.integer, np.floating)):
#                         json_serializable[key][sub_key] = float(sub_value)
#                     else:
#                         json_serializable[key][sub_key] = sub_value
#             else:
#                 json_serializable[key] = value
        
#         json.dump(json_serializable, f, indent=2)

#     print(f"\nDetailed metrics saved to {metrics_path}")
#     print("="*60)

#     return final_metrics

from src.evaluation.metrics import ClassificationMetrics
if __name__ == "__main__":
    device = torch.device(config.DEVICE)
    class_names = config.CLASS_NAMES

    # Initialise metrics
    metrics = ClassificationMetrics(num_classes=3, class_names=class_names, device=device)

    # Evaluation
    print("Metrics module loaded successfully!")
    print("Available functions:")
    print("- ClassificationMetrics: Calculate and visualize classification metrics")
    print("- TrainingTracker: Track training progress over epochs")
    print("- evaluate_model: Comprehensive model evaluation")



