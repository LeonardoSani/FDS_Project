import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

class ResNet18Binary(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Binary, self).__init__()
        
        # Load the model with ImageNet weights
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        # ADAPT INPUT: Change Conv1 from 3 channels to 1 channel
        original_conv1_weights = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            # Average the 3 RGB channels into 1 to keep pretrained knowledge
            self.resnet.conv1.weight.data = torch.mean(original_conv1_weights, dim=1, keepdim=True)

        # ADAPT OUTPUT: Change FC layer for Binary Classification (1 output)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1) 

    def forward(self, x):
        return self.resnet(x)

# train 
def fine_tune_resnet(model, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=32, lr=1e-4, weight_decay=1e-4, device=None, save_path=None):
    """
    Fine-tunes the ResNet model for binary classification.
    
    Args:
        model: The ResNet model instance.
        X_train: Training images (numpy array).
        Y_train: Training labels (numpy array).
        X_val: Validation images (numpy array).
        Y_val: Validation labels (numpy array).
        epochs: Number of epochs to train.
        batch_size: Batch size for DataLoader.
        lr: Learning rate for Adam optimizer.
        weight_decay: Weight decay for optimizer.
        device: 'cuda' or 'cpu'. If None, automatically detects.
        save_path: Path to save the best model weights.
        
    Returns:
        history: Dictionary containing metrics lists.
        model: The model with best validation F1 score loaded.
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    # Convert to tensors
    if X_train.ndim == 3:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    else:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training Loop
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_recall': [], 'train_prec': [], 'val_recall': [], 'val_prec': [], 'val_f1': []}
    best_val_f1 = 0.0
    best_model_state = None
    
    print(f"Starting ResNet fine-tuning on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_preds_all = []
        train_targets_all = []
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == targets.bool()).sum().item()
            
            train_preds_all.extend(preds.cpu().numpy().astype(int))
            train_targets_all.extend(targets.cpu().numpy().astype(int))
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        train_recall = recall_score(train_targets_all, train_preds_all, zero_division=0)
        train_prec = precision_score(train_targets_all, train_preds_all, zero_division=0)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['train_recall'].append(train_recall)
        history['train_prec'].append(train_prec)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_preds_all = []
        val_targets_all = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == targets.bool()).sum().item()
                
                val_preds_all.extend(preds.cpu().numpy().astype(int))
                val_targets_all.extend(targets.cpu().numpy().astype(int))
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        val_recall = recall_score(val_targets_all, val_preds_all, zero_division=0)
        val_prec = precision_score(val_targets_all, val_preds_all, zero_division=0)
        val_f1 = f1_score(val_targets_all, val_preds_all, zero_division=0)
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_recall'].append(val_recall)
        history['val_prec'].append(val_prec)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Recall: {val_recall:.4f}, Val Prec: {val_prec:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model based on val F1 (maximizing F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_model_state, save_path)
                print(f"Saved best model to {save_path} (F1: {best_val_f1:.4f})")
    
    print("ResNet fine-tuning complete!")
    
    if best_model_state is not None:
        print(f"Restoring best model from epoch with val F1: {best_val_f1:.4f}")
        model.load_state_dict(best_model_state)
    
    return history, model

# test dandogli il train 
def evaluate_resnet(model, X_test, Y_test, device=None, plot_results=False, model_name="Model"):
    """
    Evaluates the ResNet model on test data and computes accuracy, F1 score, recall, and precision.
    Optionally plots Confusion Matrix and ROC Curve.
    
    Args:
        model: The trained ResNet model.
        X_test: Test images (numpy array).
        Y_test: Test labels (numpy array).
        device: 'cuda' or 'cpu'. If None, automatically detects.
        plot_results: Boolean, if True plots Confusion Matrix and ROC Curve.
        model_name: String, name of the model for plot titles.
        
    Returns:
        metrics: Dictionary containing 'accuracy', 'f1', 'recall', 'precision'.
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # Convert to tensors
    if X_test.ndim == 3:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    else:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    
    # Create DataLoader
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probs = torch.sigmoid(outputs).squeeze()
            preds = probs > 0.5
            
            all_probs.extend(probs.cpu().numpy().astype(float))
            all_preds.extend(preds.cpu().numpy().astype(int))
            all_targets.extend(targets.cpu().numpy().astype(int))
    
    # Compute metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }
    
    print(f"Test Metrics ({model_name}) - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    
    if plot_results:
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Defect'])
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(all_targets, all_probs)
        
        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot CM
        disp.plot(ax=axs[0], cmap='Blues', values_format='d')
        axs[0].set_title(f'Confusion Matrix - {model_name}')
        
        # Plot ROC
        axs[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        axs[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_ylim([0.0, 1.05])
        axs[1].set_xlabel('False Positive Rate')
        axs[1].set_ylabel('True Positive Rate')
        axs[1].set_title(f'ROC Curve - {model_name}')
        axs[1].legend(loc="lower right")

        # Plot Precision-Recall
        axs[2].plot(recall_curve, precision_curve, color='purple', lw=2, label='Precision-Recall curve')
        axs[2].set_xlim([0.0, 1.0])
        axs[2].set_ylim([0.0, 1.05])
        axs[2].set_xlabel('Recall')
        axs[2].set_ylabel('Precision')
        axs[2].set_title(f'Precision-Recall Curve - {model_name}')
        axs[2].legend(loc="lower left")
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return metrics






