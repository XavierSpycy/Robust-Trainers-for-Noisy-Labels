import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_metrics(model, data_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Calculate the accuracy, precision, recall, and F1 score of the model.

    Parameters:
    model (torch.nn.Module): The model to evaluate.
    data_loader (torch.utils.data.DataLoader): The data loader.
    device (torch.device): The device to use.

    Returns:
    acc (float): The accuracy.
    precision (float): The precision.
    recall (float): The recall.
    f1 (float): The F1 score.
    """
    # Set the model to evaluation mode
    model.eval()
    # Initialize the lists to store the predictions and labels
    all_preds = []
    all_labels = []
    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over the data loader
        for _, images, labels in data_loader:
            images, labels = images.float().to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy()) # Convert to numpy array
            all_labels.extend(labels.cpu().numpy()) # Convert to numpy array
    # Convert the lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # Calculate the metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    # Set the model back to training mode
    model.train()
    return acc, precision, recall, f1