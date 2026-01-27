import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from typing import Dict, List

def print_evaluation_results(metrics: Dict, class_names: List[str]):
    """Print detailed evaluation results"""
    
    print(f"FINAL EVALUATION RESULTS")
    print("-" * 50)
    
    # Overall metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics.get('precision_macro', 0):.4f}")
    print(f"Recall (Macro): {metrics.get('recall_macro', 0):.4f}")
    print(f"F1 Score (Macro): {metrics.get('f1_macro', 0):.4f}")
    
    if 'roc_auc_macro' in metrics:
        print(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    
    if 'roc_auc_weighted' in metrics:
        print(f"ROC AUC (Weighted): {metrics['roc_auc_weighted']:.4f}")
    
    # Per-class metrics
    if 'per_class' in metrics:
        print(f"\n🎯 PER-CLASS METRICS")
        print("-" * 30)
        
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1 Score: {class_metrics['f1']:.4f}")
            print(f"  Support: {class_metrics['support']} samples")
    
    # Classification report
    print(f"DETAILED CLASSIFICATION REPORT")
    print("-" * 40)
    print(classification_report(
        metrics.get('targets', []), 
        metrics.get('predictions', []), 
        target_names=class_names,
        digits=4
    ))

def save_evaluation_results(metrics: Dict, class_names: List[str], save_dir: str, 
                          metrics_calculator=None):
    """Save detailed evaluation results to files, including the metrics calculator object"""
    
    # Create results directory if it doesn't exist
    # os.makedirs(save_dir, exist_ok=True)
    
    # Save as JSON (maintains current functionality)
    results_path = os.path.join(save_dir, 'detailed_evaluation_results.json')
    
    # Convert numpy arrays to lists for JSON serialisation
    json_results = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, dict):
            json_results[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (np.integer, np.floating)):
                    json_results[key][sub_key] = float(sub_value)
                else:
                    json_results[key][sub_key] = sub_value
        else:
            json_results[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save metrics calculator object as pickle (new functionality)
    if metrics_calculator is not None:
        import pickle
        calculator_path = os.path.join(save_dir, 'metrics_calculator.pkl')
        with open(calculator_path, 'wb') as f:
            pickle.dump(metrics_calculator, f)
        print(f"Metrics calculator saved to: {calculator_path}")
    
    # Save detailed text report (maintains current functionality)
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Model Architecture: ResNet18\n")
        f.write(f"Number of Classes: {len(class_names)}\n")
        f.write(f"Classes: {', '.join(class_names)}\n\n")
        
        f.write("OVERALL PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (Macro): {metrics.get('precision_macro', 0):.4f}\n")
        f.write(f"Recall (Macro): {metrics.get('recall_macro', 0):.4f}\n")
        f.write(f"F1 Score (Macro): {metrics.get('f1_macro', 0):.4f}\n")
        
        if 'roc_auc_macro' in metrics:
            f.write(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}\n")
        
        f.write("\nPER-CLASS PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        
        if 'per_class' in metrics:
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(f"\n{class_name}:\n")
                for metric_name, value in class_metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
        
        f.write(f"\nTotal samples evaluated: {len(metrics.get('targets', []))}\n")
        if 'predictions' in metrics:
            correct = np.sum(np.array(metrics['predictions']) == np.array(metrics['targets']))
            f.write(f"Correct predictions: {correct}\n")
            f.write(f"Incorrect predictions: {len(metrics['targets']) - correct}\n")
    
    print(f"Evaluation results saved to: {save_dir}")
    print(f"  - JSON results: {results_path}")
    print(f"  - Text report: {report_path}")
    if metrics_calculator is not None:
        print(f"  - Metrics calculator: {calculator_path}")


def print_evaluation_summary(metrics: Dict, class_names: List[str]):
    """Print final evaluation summary"""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    
    print(f"Model Architecture: ResNet18")
    print(f"Number of Classes: {len(class_names)}")
    print(f"Classes: {', '.join(class_names)}")
    
    print(f"PERFORMANCE METRICS:")
    print(f"   Accuracy: {metrics['accuracy']:.2f}%")
    print(f"   Precision (Macro): {metrics.get('precision_macro', 0):.2f}%")
    print(f"   Recall (Macro): {metrics.get('recall_macro', 0):.2f}%")
    print(f"   F1 Score (Macro): {metrics.get('f1_macro', 0):.2f}%")
    
    if 'roc_auc_macro' in metrics:
        print(f"   ROC AUC (Macro): {metrics['roc_auc_macro']:.2f}")
    
    if 'per_class' in metrics:
        best_class = max(metrics['per_class'].items(), key=lambda x: x['f1'])
        worst_class = min(metrics['per_class'].items(), key=lambda x: x['f1'])
        
        print(f"Best Performing Class: {best_class} (F1: {best_class['f1']:.2f})")
        print(f"Worst Performing Class: {worst_class} (F1: {worst_class['f1']:.2f})")
    
    print(f"Model evaluation completed successfully!")
    print("="*60)
    
def save_predictions_analysis(predictions_df: pd.DataFrame, analysis_path: str):
    """Save predictions analysis to file"""
    
    correct_predictions = predictions_df[predictions_df['correct'] == True]
    incorrect_predictions = predictions_df[predictions_df['correct'] == False]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    
    with open(analysis_path, 'w') as f:
        f.write("PREDICTIONS ANALYSIS\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total predictions: {len(predictions_df)}\n")
        f.write(f"Correct predictions: {len(correct_predictions)} ({len(correct_predictions)/len(predictions_df)*100:.1f}%)\n")
        f.write(f"Incorrect predictions: {len(incorrect_predictions)} ({len(incorrect_predictions)/len(predictions_df)*100:.1f}%)\n\n")
        
        if len(incorrect_predictions) > 0:
            f.write("ERROR ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            
            # Analyze most common errors
            error_analysis = incorrect_predictions.groupby(['true_class', 'predicted_class']).size().sort_values(ascending=False)
            f.write("\nMost common errors:\n")
            for error_pair, count in error_analysis.head(10).items():
                true_class, predicted_class = error_pair # pyright: ignore[reportGeneralTypeIssues]
                f.write(f"  {true_class} → {predicted_class}: {count} times\n")
            
            # Confidence analysis
            f.write(f"\nAverage confidence for correct predictions: {correct_predictions['confidence'].mean():.3f}\n")
            f.write(f"Average confidence for incorrect predictions: {incorrect_predictions['confidence'].mean():.3f}\n")
            
            # Add more detailed analysis
            f.write("\nDETAILED ERROR BREAKDOWN:\n")
            f.write("-" * 25 + "\n")
            
            # Show examples of errors
            f.write("\nExample incorrect predictions:\n")
            for idx, row in incorrect_predictions.head(5).iterrows():
                f.write(f"  True: {row['true_class']} (label: {row['true_label']}), ")
                f.write(f"Predicted: {row['predicted_class']} (label: {row['predicted_label']}), ")
                f.write(f"Confidence: {row['confidence']:.3f}\n")
        
        # Overall statistics
        f.write(f"\nOVERALL STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average confidence: {predictions_df['confidence'].mean():.3f}\n")
        f.write(f"Min confidence: {predictions_df['confidence'].min():.3f}\n")
        f.write(f"Max confidence: {predictions_df['confidence'].max():.3f}\n")
        
        # Per-class accuracy
        f.write(f"\nPER-CLASS ACCURACY:\n")
        f.write("-" * 20 + "\n")
        class_accuracy = predictions_df.groupby('true_class')['correct'].mean()
        for class_name, accuracy in class_accuracy.items():
            f.write(f"{class_name}: {accuracy:.3f} ({accuracy*100:.1f}%)\n")
