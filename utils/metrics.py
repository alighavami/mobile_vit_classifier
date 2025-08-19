from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_metrics(y_true, y_pred, num_classes):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred, average='macro'),
        "ROC-AUC": roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
    }
