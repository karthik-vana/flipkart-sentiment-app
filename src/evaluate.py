import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os

def get_metrics(y_true, y_pred, y_pred_prob=None):
    """
    Calculate metrics for binary classification.
    """
    metrics = {
        "f1_score": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }
    return metrics

def save_confusion_matrix(y_true, y_pred, filename):
    """
    Generate and save confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_classification_report(y_true, y_pred, filename):
    """
    Save classification report to a text file.
    """
    report = classification_report(y_true, y_pred)
    with open(filename, "w") as f:
        f.write(report)
