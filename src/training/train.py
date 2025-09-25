import json
import joblib
from src.config import config
from src.data_processing import preprocessor
from src.training import trainer
from src.evaluation import evaluator

def run_training_pipeline():
    """Orchestrates the model training and artifact saving process."""
    print("="*70)
    print("STARTING GBV PREDICTION MODEL TRAINING PIPELINE")
    print("="*70)

    # 1. Preprocess data
    X_train, X_test, y_train, y_test, encoders, scalers = preprocessor.preprocess_for_training()

    # 2. Handle class imbalance
    sample_weight = trainer.get_sample_weights(y_train)

    # 3. Train model
    model = trainer.train_model(X_train, y_train, sample_weight)
    
    # 4. Evaluate and find optimal threshold
    evaluator.evaluate_model(model, X_test, y_test, threshold=0.5) # Evaluate at default
    optimal_threshold = evaluator.find_optimal_threshold(model, X_test, y_test)
    print("\n--- Performance at Optimal Threshold ---")
    evaluator.evaluate_model(model, X_test, y_test, threshold=optimal_threshold)

    # 5. Save all artifacts
    print("\n💾 Saving artifacts...")
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(encoders, config.ENCODERS_PATH)
    joblib.dump(scalers, config.SCALERS_PATH)
    
    with open(config.TOP_FEATURES_PATH, 'w') as f:
        json.dump(config.TOP_FEATURES, f)
        
    with open(config.OPTIMAL_THRESHOLD_PATH, 'w') as f:
        json.dump({'optimal_threshold': optimal_threshold}, f)
        
    print(f"   ✅ Model saved to: {config.MODEL_PATH}")
    print(f"   ✅ Encoders saved to: {config.ENCODERS_PATH}")
    print(f"   ✅ Scalers saved to: {config.SCALERS_PATH}")
    print(f"   ✅ Top features list saved to: {config.TOP_FEATURES_PATH}")
    print(f"   ✅ Optimal threshold saved to: {config.OPTIMAL_THRESHOLD_PATH}")
    
    print("="*70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    run_training_pipeline()