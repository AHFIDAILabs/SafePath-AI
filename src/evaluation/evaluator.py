import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, recall_score, confusion_matrix, classification_report

def find_optimal_threshold(model, X_test, y_test):
    """Finds the optimal probability threshold to maximize sensitivity."""
    print("🎯 Optimizing prediction threshold for sensitivity...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Youden's J statistic to find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"✅ Optimal threshold found: {optimal_threshold:.4f}")

    return optimal_threshold

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluates the model and prints key performance metrics, including
    specificity, confusion matrix, and classification report.
    """
    print("\n📈 Evaluating model performance...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate key metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)  # Recall is sensitivity
    
    # Calculate Specificity
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f"    - Accuracy: {accuracy:.4f}")
    print(f"    - Sensitivity (Recall): {sensitivity:.4f} [CRITICAL]")
    print(f"    - Specificity: {specificity:.4f}")
    
    print("\n🔬 Confusion Matrix:")
    print(cm)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }