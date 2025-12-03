import torch
import copy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
    
    best_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    history = {'train_loss': [], 'val_f1': []} # Simplified for brevity
    
    print(f"Training on {device} for {params['epochs']} epochs...")
    
    for epoch in range(params['epochs']):
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_metrics['f1'])
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_wts = copy.deepcopy(model.state_dict())
            
        if (epoch + 1) % 5 == 0:
            print(f"Ep {epoch+1}: TrLoss {train_loss:.4f} | Val F1 {val_metrics['f1']:.4f}")
            
    model.load_state_dict(best_wts)
    return model, history