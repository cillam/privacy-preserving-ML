import json
import numpy as np
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from simple_mia import run_simple_mia
from simple_mia import compute_mia_scores, run_simple_threshold_attack
from sklearn.metrics import f1_score

class MembershipInferenceCallback(Callback):
    def __init__(self, train_inputs, train_labels, val_inputs, val_labels, run_epochs=[5, 10]):
        super().__init__()
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.val_inputs = val_inputs
        self.val_labels = val_labels
        self.run_epochs = set(run_epochs)
        self.attack_results = []
        self.epoch_predictions = []  # store predictions from every epoch
        self.f1_per_epoch = []       # new: store f1 per epoch

    def on_epoch_end(self, epoch, logs=None):
        # Predict on train and validation
        train_logits = self.model.predict(self.train_inputs, verbose=0)
        val_logits = self.model.predict(self.val_inputs, verbose=0)

        # Convert logits to probabilities
        train_probs = tf.nn.softmax(train_logits, axis=-1).numpy()
        val_probs = tf.nn.softmax(val_logits, axis=-1).numpy()

        # Save all predictions for future analysis
        self.epoch_predictions.append({
            'epoch': epoch + 1,
            'train_probs': train_probs,
            'val_probs': val_probs,
        })

        # Compute and store F1 score
        val_preds = np.argmax(val_probs, axis=1)
        f1 = f1_score(self.val_labels, val_preds, average='weighted')
        self.f1_per_epoch.append((epoch + 1, f1))

        # Only run MIA on selected epochs
        if (epoch + 1) in self.run_epochs:
            train_scores = compute_mia_scores(train_probs, self.train_labels)
            val_scores = compute_mia_scores(val_probs, self.val_labels)

            attack = run_simple_threshold_attack(train_scores, val_scores)
            self.attack_results.append((epoch + 1, attack))
            print(f"\n🔐 MIA (Epoch {epoch + 1}): Accuracy={attack['accuracy']:.4f}, AUC={attack['auc']:.4f}")

