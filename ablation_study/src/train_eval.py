import pandas as pd
import numpy as np
import torch
import sys
import os
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def safe_tqdm(*args, **kwargs):
    try:
        from tqdm.notebook import tqdm as notebook_tqdm
        return notebook_tqdm(*args, **kwargs)
    except ImportError:
        from tqdm import tqdm as std_tqdm
        return std_tqdm(*args, **kwargs)


def eval(model, data_loader, loss_fn, eval_batch, save_prediction, prediction_saving_path, active_branches, device, tqdm_position=0):
    model.eval()
    pred_csv = pd.DataFrame(columns=["file_name", "channel", "abnormality_type", "label_actual", "label_predicted", "predicted_prob"])
    
    if save_prediction:
        file_names = []
        prob_predicted = []
        label_predicted = []
        label_actual = []
        channel = []
        abnormality_type = []
        data = []

    eval_loss = 0.0
    correct_eval = 0.0
    correct_class_0 = 0.0
    correct_class_1 = 0.0
    correct_class_2 = 0.0
    final_labels = []
    final_predictions = []

    with torch.no_grad():
        for i, (file_name, x1, x2, x3, y) in enumerate(safe_tqdm(data_loader, leave=True, position=tqdm_position, desc="Eval", disable=False)):
            
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            label = y.to(device).long()

            # Fix for 1D mode backward compatibility in the original script
            x1_f = x1.view(x1.size(0), x1.size(2), -1)
            x2_f = x2.view(x2.size(0), x2.size(2), -1)
            x3_f = x3.view(x3.size(0), x3.size(2), -1)

            # ConvParallelEEG1DModel takes flattened inputs
            output, aux_output = model([x1_f, x2_f, x3_f], active_branch_indices=active_branches)
            preds = torch.argmax(output, dim=1)

            if save_prediction:
                for j in range(len(file_name)):
                    pred_probs = output[j].cpu().tolist()
                    actual_label = label[j].cpu().item()
                    predicted_label = preds[j].cpu().item()
                    
                    file_names.append(file_name[j])
                    file_parts = file_name[j].split('_')
                    prob_predicted.append(pred_probs)
                    label_predicted.append(predicted_label)
                    label_actual.append(actual_label)
                    channel.append(file_parts[-2])
                    abnormality_type.append(file_parts[-1].split(".")[0])
                    data.append(x1[j].cpu().numpy())

            loss = loss_fn(output, label)
            eval_loss += loss.item()

            correct_class_0 += ((preds == 0) & (label == 0)).sum().item()
            correct_class_1 += ((preds == 1) & (label == 1)).sum().item()
            correct_class_2 += ((preds == 2) & (label == 2)).sum().item()
            correct_eval += (preds == label).sum().item()

            final_labels.extend(label.cpu().tolist())
            final_predictions.extend(preds.cpu().tolist())

            if (i + 1) % 500 == 0:
                sys.stdout.flush()

    if save_prediction:
        pred_csv = pd.DataFrame({
            "file_name": file_names,
            "channel": channel,
            "abnormality_type": abnormality_type,
            "label_actual": label_actual,
            "label_predicted": label_predicted,
            "predicted_prob": prob_predicted,
            "data": data
        })
        os.makedirs(os.path.dirname(prediction_saving_path), exist_ok=True)
        pred_csv.to_csv(prediction_saving_path, index=False)

    precision = precision_score(final_labels, final_predictions, average='macro', zero_division=0)
    recall = recall_score(final_labels, final_predictions, average='macro', zero_division=0)
    
    # Sensitivity is effectively Recall (in multi-class, macro recall)
    sensitivity = recall 
    
    cm = confusion_matrix(final_labels, final_predictions, labels=[0, 1, 2])
    
    # Calculate macro Specificity from the multiclass confusion matrix
    specificity = 0.0
    for i in range(3):
        true_negatives = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        false_positives = np.sum(cm[:, i]) - cm[i, i]
        if (true_negatives + false_positives) > 0:
            specificity += true_negatives / (true_negatives + false_positives)
    specificity /= 3.0

    f1 = f1_score(final_labels, final_predictions, average='macro', zero_division=0)

    print(f"Correct classified: {correct_eval} / {len(data_loader.dataset)}")
    print(f"Class-wise correct: Class 0 = {correct_class_0}, Class 1 = {correct_class_1}, Class 2 = {correct_class_2}")
    print(f"Precision: {precision*100:.2f}%  Sensitivity (Recall): {sensitivity*100:.2f}%  Specificity: {specificity*100:.2f}%  F1: {f1*100:.2f}%")

    eval_loss /= len(data_loader)
    eval_acc = correct_eval / len(data_loader.dataset)

    return eval_loss, eval_acc


def train(model, data_loader, optimizer, loss_fn, active_branches, device, tqdm_position=0):
    model.train()
    train_loss = 0.0
    correct_train = 0.0
    correct_class_0 = 0.0
    correct_class_1 = 0.0
    correct_class_2 = 0.0

    for i, (z, x1, x2, x3, y) in enumerate(safe_tqdm(data_loader, leave=True, position=tqdm_position, desc="Training", disable=False)):
        optimizer.zero_grad()

        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        label = y.to(device).long()

        x1_f = x1.view(x1.size(0), x1.size(2), -1)
        x2_f = x2.view(x2.size(0), x2.size(2), -1)
        x3_f = x3.view(x3.size(0), x3.size(2), -1)

        output, aux_output = model([x1_f, x2_f, x3_f], active_branch_indices=active_branches)
        preds = torch.argmax(output, dim=1)

        correct_class_0 += ((preds == 0) & (label == 0)).sum().item()
        correct_class_1 += ((preds == 1) & (label == 1)).sum().item()
        correct_class_2 += ((preds == 2) & (label == 2)).sum().item()
        correct_train += (preds == label).sum().item()

        main_loss = loss_fn(output, label)
        aux_loss = sum(loss_fn(aux, label) for aux in aux_output)
        total_loss = main_loss + 0.3 * aux_loss

        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()

        if (i + 1) % 100 == 0:
            sys.stdout.flush()

    total_samples = len(data_loader.dataset)
    print(f"Correct Classified: {correct_train} / {total_samples}")
    print(f"Class-wise correct: Class 0 = {correct_class_0}, Class 1 = {correct_class_1}, Class 2 = {correct_class_2}")

    train_loss /= len(data_loader)
    train_acc = correct_train / total_samples

    return train_loss, train_acc


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] Counter: {self.counter} out of {self.patience}")
                sys.stdout.flush()
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def run_experiment(model, train_loader, val_loader, test_loader, active_branches, 
                   model_save_path, pred_save_path, epochs=50, device=None):
    import torch.nn as nn
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9,0.98), eps=1e-9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    best_acc = 0.0
    saved_epoch = 0
    result = {'epoch':[], 'train_loss':[], 'train_acc':[], 'eval_loss':[], 'eval_acc':[]}

    print(f'{"Epoch":>6} | {"TrainLoss":>9} | {"TrainAcc":>8} | {"EvalLoss":>8} | {"EvalAcc":>7}')
    print('-' * 55)

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, active_branches, device)
        eval_loss, eval_acc = eval(model, test_loader, loss_fn, test_loader.batch_size, False, '', active_branches, device)

        scheduler.step(eval_loss)
        early_stopping(eval_loss)
        
        result['epoch'].append(epoch+1)
        result['train_loss'].append(train_loss); result['train_acc'].append(train_acc)
        result['eval_loss'].append(eval_loss);   result['eval_acc'].append(eval_acc)

        m = ''
        if eval_acc > best_acc:
            best_acc = eval_acc
            saved_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            m = ' <-- saved'

        print(f'{epoch+1:>6} | {train_loss:>9.4f} | {train_acc:>8.4f} | {eval_loss:>8.4f} | {eval_acc:>7.4f} {m}')

        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

    # Evaluate on test set
    model.load_state_dict(torch.load(model_save_path))
    final_loss, final_acc = eval(model, test_loader, loss_fn, test_loader.batch_size, True, pred_save_path, active_branches, device)
    print(f'\\nFinal Test Performance (from best ckpt epoch {saved_epoch}) | Loss: {final_loss:.4f} | Acc: {final_acc:.4f}')
    return result

def evaluate_on_test_set(model, test_loader, active_branches, model_save_path, pred_save_path, device=None):
    """
    Evaluates a saved model on the test set without needing to retrain.
    """
    import torch.nn as nn
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Successfully loaded model weights from {model_save_path}")
    else:
        print(f"Warning: Model weights not found at {model_save_path}. Evaluating with current weights.")
        
    print(f"\\nEvaluating on test set...")
    final_loss, final_acc = eval(model, test_loader, loss_fn, test_loader.batch_size, True, pred_save_path, active_branches, device)
    print(f'\\nStandalone Final Test Performance | Loss: {final_loss:.4f} | Acc: {final_acc:.4f}')
    return final_loss, final_acc
