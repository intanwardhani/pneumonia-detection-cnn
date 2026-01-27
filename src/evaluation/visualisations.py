import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from typing import Dict, List, Tuple
from ..training.metrics import ClassificationMetrics

def create_evaluation_plots(final_metrics: Dict, metrics_calculator, class_names: List[str], save_dir: str):
    """
    Create comprehensive evaluation plots using outputs from evaluate_model_comprehensive()
    
    Args:
        final_metrics: Dictionary of final metrics from evaluate_model_comprehensive()
        metrics_calculator: ClassificationMetrics object from evaluate_model_comprehensive()
        class_names: List of class names
        save_dir: Directory to save plots
    """
    
    # Create evaluation directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics_calculator, class_names, cm_path)  # Use metrics_calculator instead of metrics
    
    # 2. ROC Curve
    roc_path = os.path.join(plots_dir, 'roc_curve.png')
    plot_roc_curve(metrics_calculator, class_names, roc_path)  # Use metrics_calculator instead of metrics
    
    # 3. Additional Analysis Plots
    create_comprehensive_analysis(metrics_calculator, class_names, plots_dir)  # Use metrics_calculator
    
    print(f"Evaluation plots saved to: {plots_dir}")

def plot_confusion_matrix(metrics_calculator, class_names: List[str], save_path: str):
    """Plot confusion matrix using metrics_calculator"""
    
    # Get confusion matrix from metrics_calculator
    if hasattr(metrics_calculator, 'confusion_matrix'):
        cm = metrics_calculator.confusion_matrix
    elif hasattr(metrics_calculator, 'get_confusion_matrix'):
        cm = metrics_calculator.get_confusion_matrix()
    else:
        raise ValueError("Metrics calculator does not have confusion matrix data")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(metrics_calculator, class_names: List[str], save_path: str):
    """Plot ROC curve using metrics_calculator"""
    
    # Get ROC data from metrics_calculator
    if hasattr(metrics_calculator, 'roc_data'):
        roc_data = metrics_calculator.roc_data
    elif hasattr(metrics_calculator, 'get_roc_data'):
        roc_data = metrics_calculator.get_roc_data()
    else:
        raise ValueError("Metrics calculator does not have ROC data")
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        if i < len(roc_data):
            fpr, tpr, auc = roc_data[i]
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
    
    plt.plot([], [], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_analysis(metrics_calculator, class_names: List[str], save_dir: str):
    """Create comprehensive analysis plots"""
    
    import matplotlib.pyplot as plt
    
    # 1. Per-class metrics bar chart
    plt.figure(figsize=(12, 6))
    
    if hasattr(metrics_calculator, 'per_class_metrics'):
        per_class_metrics = metrics_calculator.per_class_metrics
        classes = list(per_class_metrics.keys())
        precision = [per_class_metrics[c].get('precision', 0) for c in classes]
        recall = [per_class_metrics[c].get('recall', 0) for c in classes]
        f1 = [per_class_metrics[c].get('f1_score', 0) for c in classes]
        
        x = range(len(classes))
        width = 0.25
        
        plt.bar([i - width for i in x], precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar([i + width for i in x], f1, width, label='F1-Score')
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        per_class_path = os.path.join(save_dir, 'per_class_metrics.png')
        plt.savefig(per_class_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Confidence distribution
    if hasattr(metrics_calculator, 'confidences'):
        confidences = metrics_calculator.confidences
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        confidence_path = os.path.join(save_dir, 'confidence_distribution.png')
        plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Class distribution
    if hasattr(metrics_calculator, 'class_distribution'):
        class_dist = metrics_calculator.class_distribution
        
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, class_dist.values())
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution in Test Set')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        class_dist_path = os.path.join(save_dir, 'class_distribution.png')
        plt.savefig(class_dist_path, dpi=300, bbox_inches='tight')
        plt.close()

def get_best_class(metrics: Dict, class_names: List[str]) -> str:
    """Get the best performing class"""
    if 'per_class' not in metrics:
        return "N/A"
    
    best_class = max(metrics['per_class'].items(), key=lambda x: x['f1'])
    return f"{best_class} (F1: {best_class['f1']:.3f})"

def get_worst_class(metrics: Dict, class_names: List[str]) -> str:
    """Get the worst performing class"""
    if 'per_class' not in metrics:
        return "N/A"
    
    worst_class = min(metrics['per_class'].items(), key=lambda x: x['f1'])
    return f"{worst_class} (F1: {worst_class['f1']:.3f})"
