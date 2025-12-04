import torch
import copy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
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

def train_model(model, train_loader, val_loader, params, device):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  factor=params.get('scheduler_factor', 0.5),
                                  patience=params.get('scheduler_patience', 5))

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=params.get('early_stopping_patience', 10),
                                   min_delta=params.get('early_stopping_min_delta', 0.001),
                                   mode='max', verbose=False)

    best_wts = copy.deepcopy(model.state_dict())
    best_f1 = float('-inf')
    history = {'train_loss': [], 'val_f1': [], 'val_loss': []}

    print(f"Training on {device} for {params['epochs']} epochs...")

    for epoch in range(params['epochs']):
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        current_val_f1 = val_metrics['f1']

        history['train_loss'].append(train_loss)
        history['val_f1'].append(current_val_f1)
        history['val_loss'].append(val_metrics['loss'])

        # Check for best model based on validation F1
        if current_val_f1 > best_f1:
            best_f1 = current_val_f1
            best_wts = copy.deepcopy(model.state_dict())

        # Step the learning rate scheduler
        scheduler.step(current_val_f1)

        # Check for early stopping
        if early_stopping(current_val_f1):
            print(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {early_stopping.patience} epochs).")
            break

        print(f"Ep {epoch+1}: TrLoss {train_loss:.4f} | Val F1 {current_val_f1:.4f} | ValLoss {val_metrics['loss']:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}")

    model.load_state_dict(best_wts)
    return model, history