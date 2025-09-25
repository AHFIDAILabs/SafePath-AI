# src/training/trainer.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from src.config import config

def get_sample_weights(y_train):
    """Computes and returns sample weights to handle class imbalance."""
    print("⚖️ Computing class weights for imbalance handling...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    sample_weight = np.array([class_weight_dict[class_val] for class_val in y_train])
    print("✅ Sample weights computed.")
    return sample_weight

def train_model(X_train, y_train, sample_weight):
    """Trains the Gradient Boosting model with best hyperparameters."""
    print("🚀 Starting model training...")
    gb_model = GradientBoostingClassifier(**config.BEST_HYPERPARAMS)

    start_time = pd.Timestamp.now()
    gb_model.fit(X_train, y_train, sample_weight=sample_weight)
    end_time = pd.Timestamp.now()

    duration = (end_time - start_time).total_seconds()
    print(f"✅ Model training completed in {duration:.2f} seconds.")
    return gb_model