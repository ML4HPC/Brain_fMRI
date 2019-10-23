import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

def outputs_multi_classification_report(y_true, y_preds):
    y_pred_labels = []
    for pred in y_preds:
        sm_pred = F.log_softmax(pred, dim=1)
        pred_label = sm_pred.max(1, keepdim=True)[1]
        y_pred_labels.append(pred_label)
    
    return classification_report(y_true, y_pred_labels)

def outputs_bin_classification_report(y_true, y_preds):
    y_pred_labels = []
    for pred in y_preds:
        pred_label = torch.nn.Sigmoid(pred) > 0.5
        y_pred_labels.append(pred_label)
    
    return classification_report(y_true, y_pred_labels)

    

