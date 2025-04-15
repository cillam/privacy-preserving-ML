# Simple membership inference attack functions

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Calculates confidence scores used for membership inference
def compute_mia_scores(logits_or_probs, labels):
    """
    Expects probabilities. If model outputs logits, softmax must be applied first.
    """
    if logits_or_probs.ndim == 2 and logits_or_probs.shape[1] > 1:
        # Multiclass case — take max probability
        confidences = np.max(logits_or_probs, axis=1)
    else:
        # Binary case
        confidences = logits_or_probs.ravel()
    return confidences


# Basic threshold attack: higher confidence => more likely to be in train set
def run_simple_threshold_attack(train_scores, test_scores):

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


# Computes MIA using confidence scores + threshold attack
def run_simple_mia(train_logits_or_probs, test_logits_or_probs):

    train_scores = compute_mia_scores(train_logits_or_probs, None)
    test_scores = compute_mia_scores(test_logits_or_probs, None)

    attack_result = run_simple_threshold_attack(train_scores, test_scores)
    return attack_result
