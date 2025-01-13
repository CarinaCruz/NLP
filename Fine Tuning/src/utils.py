from sklearn.metrics import accuracy_score, f1_score
import tqdm
import torch
import numpy as np 
import time
import os

def calculate_metrics(true_labels, predictions):
    """Used to calculate metrics on training

    Args:
        true_labels (list)
        predictions (list)

    Returns:
        dict: calculated metrics
    """
    return {
        "f1": f1_score(true_labels, predictions, average='macro'),
        "accuracy": accuracy_score(true_labels, predictions)
    }

def evaluation(eval_dataloader, model, device):     
    """ Evaluate fine tuning mode training

    Args:
        eval_dataloader (DataLoader): evaluation dataset loader
        model (transformer model): model to be trained on
        device (str): define gpu or cpu device

    Returns:
        list: evaluation metrics and loss 
    """

    val_acc_loss = 0    
    true_labels = []
    pred_labels = []
    model.eval()

    for val_batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Validation", leave=False):
        with torch.no_grad():
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            val_outputs = model(**val_batch)
            val_logits = val_outputs.logits
            val_loss = val_outputs.loss
            val_acc_loss += val_loss.item()                                
            
            predictions_v = torch.argmax(val_logits, dim=1)                        
            true_labels.extend(val_batch['labels'].cpu().numpy())
            pred_labels.extend(predictions_v.cpu().numpy())

    val_mean_loss = val_acc_loss / len(eval_dataloader) if len(eval_dataloader) > 0 else 0
    metrics = calculate_metrics(true_labels, pred_labels)
    return val_mean_loss, metrics


def training_loop(train_dataloader, eval_dataloader, num_epochs, model,
                  optimizer, lr_scheduler, device, model_type='fine_tuning'):    
    """ Receives data and operate Training

    Args:
        train_dataloader (DataLoader): training dataset loader
        eval_dataloader (DataLoader): training dataset loader
        num_epochs (int): num of iteractions
        model (transformer model): model to be trained on
        optimizer (torch.optim.AdamW): model optimizer
        lr_scheduler (transformers.get_scheduler):model learning rate config        
        device (str): define gpu or cpu device
        model_type (str, optional): _description_. Defaults to 'fine_tuning'.

    Returns:
        list: training metrics, loss and total time
    """
    os.makedirs("models", exist_ok=True)

    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
    start_train = time.time()
    best_val_f1 = 0
    early_stopping_patience = max(3, num_epochs // 2)
    early_stopping_counter = 0

    all_training_metrics, all_losses = [], []
    val_losses, val_metrics = [], []

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        all_predictions = []
        all_labels = []
        acc_loss = 0

        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}", leave=False):      
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss    
            acc_loss += loss.item()         

            # Predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            # Save predictions 
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()        
            progress_bar.update(1)                

        # Metrics for training
        metrics = calculate_metrics(all_labels, all_predictions)    
        all_training_metrics.append(metrics)

        # Mean loss
        mean_loss = acc_loss / len(train_dataloader)
        all_losses.append(mean_loss)
        print(f"epoch {epoch+1}/{num_epochs} - mean loss: {mean_loss:.4f} - f1 score: {metrics['f1']:.4f} - accuracy: {metrics['accuracy']:.4f}")        

        # Validation step
        val_loss, val_metric = evaluation(eval_dataloader, model, device)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        if val_metric['f1'] > best_val_f1:
            best_val_f1 = val_metric['f1']
            early_stopping_counter = 0
            print(f"Epoch {epoch+1}: Saving best model with F1 {val_metric['f1']:.4f}")
            if model_type == 'peft':
                torch.save(model.state_dict(), "models/peft_best_model.pt")
            else:
                torch.save(model.state_dict(), "models/best_model.pt")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    end_train = time.time()
    total_time = (end_train - start_train) / 60
    print(f"Training completed successfully in {total_time:.2f} minutes")
    return val_metrics, val_losses, all_training_metrics, all_losses, total_time
