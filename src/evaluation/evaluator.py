# Add these imports at the top:
import os
import torch
import pandas as pd
from typing import Dict, Tuple, List
from tqdm import tqdm
from .save_eval import save_evaluation_results, save_predictions_analysis
from ..training.metrics import ClassificationMetrics


def evaluate_model_comprehensive(model, dataloader, class_names, device,
                                 save_dir: str) -> Tuple[Dict, 'ClassificationMetrics']:
    """
    Comprehensive model evaluation with all metrics
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation
        class_names: List of class names
        device: Device to perform evaluation on
        save_dir: Directory to save evaluation results
    
    Returns:
        tuple: (final_metrics, metrics_calculator)
    """
    
    # Initialise metrics calculator
    num_classes = len(class_names)
    metrics_calculator = ClassificationMetrics(num_classes, class_names, device)
    
    print(f"Evaluating model on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Update metrics
            metrics_calculator.update(outputs, labels, loss=None) # type: ignore
    
    # Calculate final metrics
    final_metrics = metrics_calculator.calculate_metrics()
    
    save_evaluation_results(
        metrics=final_metrics,
        class_names=class_names,
        save_dir=save_dir,
        metrics_calculator=metrics_calculator
    )
    
    return final_metrics, metrics_calculator

def generate_predictions_file(model, dataloader, class_names: List[str], 
                              device, save_dir: str) -> str:
    """
    Generate detailed predictions file
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for predictions
        class_names: List of class names
        device: Device to perform evaluation on
        save_dir: Directory to save predictions file (will be created if it doesn't exist)
    
    Returns:
        str: Path to the saved predictions file
    """
    
    print("Generating detailed predictions...")
    
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Generating predictions")):
            images = images.to(device)
            labels = labels.to(device)  # Move labels to device as well
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            for i in range(len(images)):
                prediction = {
                    'true_label': labels[i].item(),
                    'predicted_label': predictions[i].item(),
                    'confidence': probabilities[i].max().item(),
                    'true_class': class_names[labels[i].item()],
                    'predicted_class': class_names[predictions[i].item()],      # type: ignore
                    'probabilities': probabilities[i].cpu().numpy().tolist(),
                    'correct': predictions[i].item() == labels[i].item()
                }
                all_predictions.append(prediction)
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # Create predictions directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to CSV file in the predictions directory
    predictions_file_path = os.path.join(save_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_file_path, index=False)
    
    # Save analysis
    analysis_file_path = os.path.join(save_dir, 'predictions_analysis.txt')
    save_predictions_analysis(predictions_df, analysis_file_path)
    
    print(f"Predictions saved to: {predictions_file_path}")
    print(f"Analysis saved to: {analysis_file_path}")
    
    return predictions_file_path

