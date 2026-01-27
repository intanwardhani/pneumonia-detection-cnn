import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Union
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, average_precision_score)

class ClassificationMetrics:
    """
    Visualisation class for ClassificationMetrics from evaluate_model_comprehensive()
    
    This class takes an existing metrics_calculator object and provides plotting
    functionality for ROC curves, confusion matrices, and precision-recall curves.
    """
    
    def __init__(self, metrics_calculator):
        """
        Initialise the visualiser with an existing metrics_calculator
        
        Args:
            metrics_calculator: ClassificationMetrics object from evaluate_model_comprehensive()
        """
        
        # Copy all necessary attributes from the metrics_calculator
        self.num_classes = metrics_calculator.num_classes
        self.class_names = metrics_calculator.class_names
        self.device = metrics_calculator.device
        self.predictions = metrics_calculator.predictions
        self.targets = metrics_calculator.targets
        self.probabilities = metrics_calculator.probabilities
        self.losses = metrics_calculator.losses
        
        # Calculate metrics once for consistency
        self._metrics = None
    
    def get_metrics(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Get calculated metrics (calculates if not already done)
        
        Returns:
            Dictionary containing all calculated metrics
        """
        if self._metrics is None:
            self._metrics = self._calculate_metrics()
        return self._metrics
    
    def _calculate_metrics(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate all metrics (internal method)
        
        Returns:
            Dictionary containing all calculated metrics
        """
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        if self.num_classes == 1:
            # Binary classification metrics
            metrics['precision'] = precision_score(targets, predictions, average='binary', zero_division=0)
            metrics['recall'] = recall_score(targets, predictions, average='binary', zero_division=0)
            metrics['f1'] = f1_score(targets, predictions, average='binary', zero_division=0)
            
            # ROC AUC for binary classification
            if len(np.unique(targets)) > 1:  # Need both classes for AUC
                metrics['roc_auc'] = roc_auc_score(targets, probabilities)
                metrics['average_precision'] = average_precision_score(targets, probabilities)
            else:
                metrics['roc_auc'] = 1.0  # Perfect score when only one class present
                metrics['average_precision'] = 1.0
        
        else:
            # Multi-class classification metrics
            metrics['precision_macro'] = precision_score(targets, predictions, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(targets, predictions, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
            
            metrics['precision_weighted'] = precision_score(targets, predictions, average='weighted', zero_division=0)
            metrics['recall_weighted'] = recall_score(targets, predictions, average='weighted', zero_division=0)
            metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted', zero_division=0)
            
            # ROC AUC for multi-class (one-vs-rest)
            if len(np.unique(targets)) > 1:
                metrics['roc_auc_macro'] = roc_auc_score(targets, probabilities, multi_class='ovr', average='macro')
                metrics['roc_auc_weighted'] = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
            else:
                metrics['roc_auc_macro'] = 1.0
                metrics['roc_auc_weighted'] = 1.0
        
        # Per-class metrics
        if self.num_classes > 1:
            try:
                class_report = classification_report(
                    targets, predictions, target_names=self.class_names, 
                    output_dict=True, zero_division=0
                )
                metrics['per_class'] = {}
                for i, class_name in enumerate(self.class_names):
                    metrics['per_class'][class_name] = {
                        'precision': class_report[class_name]['precision'], # type: ignore
                        'recall': class_report[class_name]['recall'],       # type: ignore
                        'f1': class_report[class_name]['f1-score'],         # type: ignore
                        'support': class_report[class_name]['support']      # type: ignore
                    }
            except Exception as e:
                logging.warning(f"Could not calculate per-class metrics: {e}")
                metrics['per_class'] = {}
        
        # Average loss
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
        
        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return confusion_matrix(targets, predictions)

    def plot_confusion_matrix(self, save_path: str, figsize: Tuple[int, int] = (10, 8)):
        """Plot confusion matrix"""
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()

    def plot_roc_curve(self, save_path: str, figsize: Tuple[int, int] = (10, 8)):
        """Plot ROC curve"""
        if self.num_classes == 1:
            # Binary ROC curve
            targets = np.array(self.targets)
            probabilities = np.array(self.probabilities)
            
            fpr, tpr, _ = roc_curve(targets, probabilities)
            roc_auc = roc_auc_score(targets, probabilities)
            
            plt.figure(figsize=figsize)
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([], [], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"ROC curve saved to {save_path}")
            
            plt.show()
        else:
            # Multi-class ROC curves
            targets = np.array(self.targets)
            probabilities = np.array(self.probabilities)
            
            plt.figure(figsize=figsize)
            
            for i, class_name in enumerate(self.class_names):
                # Binary classification for this class
                binary_targets = (targets == i).astype(int)
                binary_probs = probabilities[:, i]
                
                fpr, tpr, _ = roc_curve(binary_targets, binary_probs)
                roc_auc = roc_auc_score(binary_targets, binary_probs)
                
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
            
            plt.plot([], [], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Multi-class ROC curves saved to {save_path}")
            
            plt.show()

    def plot_precision_recall_curve(self, save_path: str, figsize: Tuple[int, int] = (10, 8)):
        """Plot Precision-Recall curve"""
        if self.num_classes == 1:
            # Binary PR curve
            targets = np.array(self.targets)
            probabilities = np.array(self.probabilities)
            
            precision, recall, _ = precision_recall_curve(targets, probabilities)
            avg_precision = average_precision_score(targets, probabilities)
            
            plt.figure(figsize=figsize)
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AP = {avg_precision:.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Precision-Recall curve saved to {save_path}")
            
            plt.show()
        else:
            logging.warning("Precision-Recall curves not implemented for multi-class yet")

    def print_summary(self):
        """Print comprehensive metrics summary"""
        metrics = self.get_metrics()
        
        print("=== MODEL PERFORMANCE SUMMARY ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

        if self.num_classes == 1:
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            if 'roc_auc' in metrics:
                print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        else:
            print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
            print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
            print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
            if 'roc_auc_macro' in metrics:
                print(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")

        # Per-class performance
        if 'per_class' in metrics:
            print("\n=== PER-CLASS PERFORMANCE ===")
            for class_name, class_metrics in metrics['per_class'].items():  # type: ignore
                print(f"{class_name}:")
                print(f"  Precision: {class_metrics['precision']:.4f}")     # type: ignore
                print(f"  Recall: {class_metrics['recall']:.4f}")           # type: ignore
                print(f"  F1 Score: {class_metrics['f1']:.4f}")             # type: ignore
                print(f"  Support: {class_metrics['support']} samples")     # type: ignore
                
    def save_summary(self, save_path: str):
        """Save metrics summary to a text file"""
        
        metrics = self.get_metrics()
        with open(save_path, 'w') as f:
            f.write("=== MODEL PERFORMANCE SUMMARY ===\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")

            if self.num_classes == 1:
                f.write("BINARY CLASSIFICATION METRICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1 Score: {metrics['f1']:.4f}\n")
                if 'roc_auc' in metrics:
                    f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
                if 'average_precision' in metrics:
                    f.write(f"Average Precision: {metrics['average_precision']:.4f}\n")
            else:
                f.write("MULTI-CLASS CLASSIFICATION METRICS:\n")
                f.write("-" * 35 + "\n")
                f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
                f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
                f.write(f"F1 Score (Macro): {metrics['f1_macro']:.4f}\n")
                if 'roc_auc_macro' in metrics:
                    f.write(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}\n")
                if 'roc_auc_weighted' in metrics:
                    f.write(f"ROC AUC (Weighted): {metrics['roc_auc_weighted']:.4f}\n")
            
            # Per-class performance
            if 'per_class' in metrics:
                f.write("\n=== PER-CLASS PERFORMANCE ===\n")
                f.write("-" * 30 + "\n")
                for class_name, class_metrics in metrics['per_class'].items():  # type: ignore
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {class_metrics['precision']:.4f}\n") # type: ignore
                    f.write(f"  Recall: {class_metrics['recall']:.4f}\n")       # type: ignore
                    f.write(f"  F1 Score: {class_metrics['f1']:.4f}\n")         # type: ignore
                    f.write(f"  Support: {class_metrics['support']} samples\n") # type: ignore
            
            # Additional info
            f.write(f"\n=== EVALUATION DETAILS ===\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total Classes: {self.num_classes}\n")
            f.write(f"Class Names: {', '.join(self.class_names)}\n")
            f.write(f"Total Samples: {len(self.predictions)}\n")
            if self.losses:
                f.write(f"Average Loss: {metrics.get('avg_loss', 'N/A'):.4f}\n")
        
        logging.info(f"Summary saved to: {save_path}")

