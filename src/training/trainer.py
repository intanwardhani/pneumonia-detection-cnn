import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from config import Config

config = Config()
device = config.DEVICE
print(f"Using device: {device}")
def setup_logging(): 
    """Setup logging configuration""" 
    
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        handlers=[logging.FileHandler('training.log'), 
                                  logging.StreamHandler()])

def train_epoch(model, train_loader, criterion, optimiser, device):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Handle different label formats
        if isinstance(criterion, nn.CrossEntropyLoss):
            labels = labels.long()
        else:
            labels = labels.float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
        
        # Forward pass
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Calculate accuracy
        if isinstance(criterion, nn.CrossEntropyLoss):
            predicted = outputs.argmax(dim=1)
        else:
            predicted = (torch.sigmoid(outputs) > 0.5).float()
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            # Handle different label formats
            if isinstance(criterion, nn.CrossEntropyLoss):
                labels = labels.long()
            else:
                labels = labels.float()
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            
            # Calculate accuracy and collect predictions
            if isinstance(criterion, nn.CrossEntropyLoss):
                predicted = outputs.argmax(dim=1)
            else:
                predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, optimiser, scheduler, class_names=None, 
                num_epochs=config.NUM_EPOCHS, patience=5, min_delta=0.001, 
                save_path='best_model.pth', device=None):
    """Main training loop with comprehensive features
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimiser: Optimiser for training
        scheduler: Learning rate scheduler
        class_names: List of class names for metrics
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        save_path: Path to save best model
        device: Device to train on (auto-detect if None)
    
    Returns:
        dict: Training results and metrics
    """
    
    # Detect device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Choose appropriate loss function
    if class_names and len(class_names) > 1:
        # Multi-class classification
        criterion = nn.CrossEntropyLoss()
    else:
        # Binary classification
        criterion = nn.BCEWithLogitsLoss()
    
    # Initialise tracking variables
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Class names: {class_names}")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, 
                                                                  val_loader, 
                                                                  criterion, 
                                                                  device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Early stopping logic
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'class_names': class_names
            }, save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    else:
        print("No improvement during training")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_loss': best_val_loss,
        'best_epoch': len(val_losses) - patience_counter if best_model_state is not None else None,
        'total_epochs': len(train_losses)
    }

