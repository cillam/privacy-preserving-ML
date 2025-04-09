# simple_mia.py
# Lightweight membership inference attack module for TensorFlow models

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def compute_mia_scores(logits_or_probs, labels):
    """
    Calculates confidence scores used for membership inference.
    Expects raw probabilities or logits (after softmax).
    """
    if logits_or_probs.ndim == 2 and logits_or_probs.shape[1] > 1:
        # Multiclass case — take max probability
        confidences = np.max(logits_or_probs, axis=1)
    else:
        # Binary case
        confidences = logits_or_probs.ravel()
    return confidences


def run_simple_threshold_attack(train_scores, test_scores):
    """
    Basic threshold attack: higher confidence => more likely to be in train set
    """
    all_scores = np.concatenate([train_scores, test_scores])
    labels = np.concatenate([np.ones_like(train_scores), np.zeros_like(test_scores)])

    thresholds = np.unique(all_scores)
    best_acc = 0
    best_threshold = 0
    for t in thresholds:
        preds = (all_scores >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    auc = roc_auc_score(labels, all_scores)
    return {
        'attack_type': 'threshold_confidence',
        'accuracy': best_acc,
        'auc': auc,
        'threshold': best_threshold,
    }


def run_logistic_regression_attack(train_features, test_features, train_labels, test_labels):
    """
    Trains a simple logistic regression attack model on confidence scores.
    """
    X = np.concatenate([train_features, test_features])
    y = np.concatenate([np.ones_like(train_labels), np.zeros_like(test_labels)])

    model = LogisticRegression(solver='liblinear')
    model.fit(X.reshape(-1, 1), y)
    preds = model.predict(X.reshape(-1, 1))
    auc = roc_auc_score(y, model.predict_proba(X.reshape(-1, 1))[:, 1])
    acc = accuracy_score(y, preds)

    return {
        'attack_type': 'logistic_regression',
        'accuracy': acc,
        'auc': auc
    }


def run_simple_mia(train_logits_or_probs, test_logits_or_probs):
    """
    Wrapper to compute MIA using confidence scores + threshold attack.
    """
    train_scores = compute_mia_scores(train_logits_or_probs, None)
    test_scores = compute_mia_scores(test_logits_or_probs, None)

    attack_result = run_simple_threshold_attack(train_scores, test_scores)
    return attack_result
