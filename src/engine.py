import torch
import copy
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
# NEW IMPORTS
from torch.utils.data import DataLoader, TensorDataset

# ... [Keep EarlyStopping class as is] ...
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='max', verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if self.mode == 'min':
            self.val_score_sign = -1
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.val_score_sign = 1
            self.best_score = float('-inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self.val_score_sign * current_score < self.val_score_sign * (self.best_score + self.min_delta):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0
        return self.early_stop

# ... [Keep train_one_epoch, validate, and train_model as is] ...
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
    return running_loss / len(loader.dataset), accuracy_score(all_targets, all_preds)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    metrics = {
        'loss': running_loss / len(loader.dataset),
        'acc': accuracy_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds),
        'recall': recall_score(all_targets, all_preds)
    }
    return metrics

def train_model(model, train_loader, val_loader, params, device, verbose=True):
    criterion = torch.nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=params['lr'],
        weight_decay=params.get('weight_decay', 0) 
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  factor=params.get('scheduler_factor', 0.5),
                                  patience=params.get('scheduler_patience', 5))

    early_stopping = EarlyStopping(patience=params.get('early_stopping_patience', 10),
                                   min_delta=params.get('early_stopping_min_delta', 0.001),
                                   mode='max', verbose=False)

    best_wts = copy.deepcopy(model.state_dict())
    best_f1 = float('-inf')
    best_metrics = {} 

    history = {'train_loss': [], 'val_f1': [], 'val_loss': []}

    if verbose:
        print(f"Training on {device} for {params['epochs']} epochs...")

    for epoch in range(params['epochs']):
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        current_val_f1 = val_metrics['f1']

        history['train_loss'].append(train_loss)
        history['val_f1'].append(current_val_f1)
        history['val_loss'].append(val_metrics['loss'])

        if current_val_f1 > best_f1:
            best_f1 = current_val_f1
            best_wts = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics

        scheduler.step(current_val_f1)

        if early_stopping(current_val_f1):
            if verbose:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
            break
        
        if verbose:
            print(f"Ep {epoch+1}: TrLoss {train_loss:.4f} | Val F1 {current_val_f1:.4f} | ValLoss {val_metrics['loss']:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}")

    model.load_state_dict(best_wts)
    return model, history, best_metrics

# ==========================================
# UPDATED TUNING FUNCTION
# ==========================================

def create_loader(X, y, batch_size, shuffle=True):
    # Ensure inputs are tensors and handle channel dim
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if X.ndim == 3: # Add channel dim if missing (N, H, W) -> (N, 1, H, W)
        X = X.unsqueeze(1)
        
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def tune_hyperparameters(model_class, X_train, Y_train, X_val, Y_val, param_grid, device, n_trials=10):
    """
    Performs Random Search. Accepts RAW DATA (X, Y) to allow dynamic batch_size creation.
    """
    print(f"--- Starting Hyperparameter Tuning ({n_trials} trials) ---")
    
    best_score = -1.0
    best_params = None
    best_model = None
    tuning_results = []

    for i in range(n_trials):
        # 1. Randomly sample parameters
        current_params = {k: random.choice(v) for k, v in param_grid.items()}
        
        # Default defaults
        if 'epochs' not in current_params: current_params['epochs'] = 5
        bs = current_params.get('batch_size', 32)
            
        print(f"\n[Trial {i+1}/{n_trials}] Params: {current_params}")

        # 2. Create Loaders with the specific Batch Size for this trial
        train_loader = create_loader(X_train, Y_train, batch_size=bs, shuffle=True)
        val_loader = create_loader(X_val, Y_val, batch_size=bs, shuffle=False)

        # 3. Instantiate fresh model
        model = model_class(num_classes=1).to(device)

        # 4. Train
        trained_model, history, final_metrics = train_model(
            model, train_loader, val_loader, current_params, device, verbose=False
        )

        # 5. Evaluate
        score = final_metrics.get('f1', 0.0)
        loss_score = final_metrics.get('loss', 99.9)
        print(f"   -> Result: Val F1: {score:.4f} (Loss: {loss_score:.4f})")

        result_entry = {
            'trial': i,
            'params': current_params,
            'metrics': final_metrics,
            'history': history
        }
        tuning_results.append(result_entry)

        if score > best_score:
            best_score = score
            best_params = current_params
            best_model = copy.deepcopy(trained_model)
            print(f"   *** New Best Model Found! ***")

    print("\n--- Tuning Complete ---")
    print(f"Best F1 Score: {best_score:.4f}")
    print(f"Best Params: {best_params}")
    
    return best_params, best_model, tuning_results