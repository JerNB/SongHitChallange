import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter
import matplotlib.pyplot as plt

# Load data
DATA_PATH = 'song_data.csv'

def analyze_class_distribution(y, title="Class Distribution"):
    """Analyze and print class distribution."""
    counter = Counter(y)
    total = len(y)
    print(f"\n{title}:")
    print(f"Non-hits (0): {counter[0]} ({counter[0]/total*100:.1f}%)")
    print(f"Hits (1): {counter[1]} ({counter[1]/total*100:.1f}%)")
    print(f"Imbalance ratio: {counter[0]/counter[1]:.2f}:1")
    return counter

def load_data(path=DATA_PATH, popularity_threshold=80):
    """Load the Spotify song dataset and create binary label 'hit' based on popularity."""
    df = pd.read_csv(path)
    df['hit'] = (df['popularity'] >= popularity_threshold).astype(int)
    
    # Analyze initial class distribution
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    print(f"Total songs in dataset: {len(df):,}")
    analyze_class_distribution(df['hit'], "Original Dataset")
    
    numeric_cols = [
        'energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence',
        'speechiness', 'instrumentalness', 'mode', 'key', 'duration_ms', 'acousticness'
    ]
    X = df[numeric_cols]
    y = df['hit']
    
    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Second split: separate train and validation from remaining data (60% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print("\nAfter train/val/test split:")
    analyze_class_distribution(y_train, "Training Set")
    analyze_class_distribution(y_val, "Validation Set")
    analyze_class_distribution(y_test, "Test Set")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_balanced_sample(X_train, y_train, balance_method='undersample'):
    """Create a balanced dataset using simple undersampling or oversampling."""
    counter = Counter(y_train)
    majority_class = max(counter, key=counter.get)
    minority_class = min(counter, key=counter.get)
    
    majority_indices = np.where(y_train == majority_class)[0]
    minority_indices = np.where(y_train == minority_class)[0]
    
    if balance_method == 'undersample':
        # Random undersample majority class
        np.random.seed(42)
        selected_majority = np.random.choice(majority_indices, size=len(minority_indices), replace=False)
        balanced_indices = np.concatenate([selected_majority, minority_indices])
        
    elif balance_method == 'oversample':
        # Random oversample minority class
        np.random.seed(42)
        oversample_size = len(majority_indices) - len(minority_indices)
        oversampled_minority = np.random.choice(minority_indices, size=oversample_size, replace=True)
        balanced_indices = np.concatenate([majority_indices, minority_indices, oversampled_minority])
    
    # Shuffle the indices
    np.random.seed(42)
    np.random.shuffle(balanced_indices)
    
    X_balanced = X_train.iloc[balanced_indices]
    y_balanced = y_train.iloc[balanced_indices]
    
    return X_balanced, y_balanced

def train_models_with_class_weights(X_train, y_train, use_class_weights=False):
    """Train multiple classifiers with optional class weight balancing."""
    models = {}
    
    # Set class weights if specified
    class_weight = 'balanced' if use_class_weights else None
    
    # Logistic Regression
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight))
    ])
    lr_pipe.fit(X_train, y_train)
    models['Logistic Regression'] = lr_pipe

    # Support Vector Machine
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42, class_weight=class_weight))
    ])
    svm_pipe.fit(X_train, y_train)
    models['SVM'] = svm_pipe

    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weight)
    rf_clf.fit(X_train, y_train)
    models['Random Forest'] = rf_clf

    # Neural Network (MLP) - doesn't support class_weight directly
    mlp_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42))
    ])
    mlp_pipe.fit(X_train, y_train)
    models['Neural Network'] = mlp_pipe

    return models

def evaluate_with_threshold_tuning(models, X_val, X_test, y_val, y_test, technique_name="Original"):
    """Evaluate models with threshold tuning for better recall."""
    results = []
    
    for name, model in models.items():
        # Get probability predictions
        if hasattr(model, 'predict_proba'):
            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]
        else:
            val_probs = model.decision_function(X_val)
            test_probs = model.decision_function(X_test)
        
        # Find optimal threshold based on validation F1 score
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, val_probs)
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
        optimal_threshold_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_threshold_idx] if optimal_threshold_idx < len(thresholds) else 0.5
        
        # Apply optimal threshold
        val_preds_optimal = (val_probs >= optimal_threshold).astype(int)
        test_preds_optimal = (test_probs >= optimal_threshold).astype(int)
        
        # Default threshold (0.5) predictions
        val_preds_default = (val_probs >= 0.5).astype(int)
        test_preds_default = (test_probs >= 0.5).astype(int)
        
        # Calculate metrics for both thresholds
        result = {
            'Model': name,
            'Optimal_Threshold': optimal_threshold,
            'Test_Accuracy_Default': accuracy_score(y_test, test_preds_default),
            'Test_Precision_Default': precision_score(y_test, test_preds_default, zero_division=0),
            'Test_Recall_Default': recall_score(y_test, test_preds_default, zero_division=0),
            'Test_F1_Default': f1_score(y_test, test_preds_default, zero_division=0),
            'Test_Accuracy_Optimal': accuracy_score(y_test, test_preds_optimal),
            'Test_Precision_Optimal': precision_score(y_test, test_preds_optimal, zero_division=0),
            'Test_Recall_Optimal': recall_score(y_test, test_preds_optimal, zero_division=0),
            'Test_F1_Optimal': f1_score(y_test, test_preds_optimal, zero_division=0)
        }
        results.append(result)
    
    # Display results
    print(f"\n{'='*110}")
    print(f"{' '*35}{technique_name} - Model Results")
    print(f"{'='*110}")
    print(f"{'Model':<20} | {'Threshold':<10} | {'Default (0.5)':<35} | {'Optimized Threshold':<35}")
    print(f"{'':<20} | {'':<10} | {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} | {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("-"*110)
    
    for result in results:
        print(f"{result['Model']:<20} | "
              f"{result['Optimal_Threshold']:<10.3f} | "
              f"{result['Test_Accuracy_Default']:<8.3f} {result['Test_Precision_Default']:<8.3f} "
              f"{result['Test_Recall_Default']:<8.3f} {result['Test_F1_Default']:<8.3f} | "
              f"{result['Test_Accuracy_Optimal']:<8.3f} {result['Test_Precision_Optimal']:<8.3f} "
              f"{result['Test_Recall_Optimal']:<8.3f} {result['Test_F1_Optimal']:<8.3f}")
    
    print("="*110)
    return results

def main():
    # Load and analyze data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    print("\n" + "="*90)
    print("COMPARING DIFFERENT CLASS BALANCING APPROACHES")
    print("="*90)
    
    all_results = {}
    
    # 1. Original data (imbalanced)
    print(f"\n*** 1. ORIGINAL IMBALANCED DATA ***")
    models_original = train_models_with_class_weights(X_train, y_train, use_class_weights=False)
    results_original = evaluate_with_threshold_tuning(models_original, X_val, X_test, y_val, y_test, "Original Imbalanced")
    all_results["Original"] = results_original
    
    # 2. Class weights (balanced)
    print(f"\n*** 2. CLASS WEIGHTS (BALANCED) ***")
    models_weighted = train_models_with_class_weights(X_train, y_train, use_class_weights=True)
    results_weighted = evaluate_with_threshold_tuning(models_weighted, X_val, X_test, y_val, y_test, "Class Weights")
    all_results["Class Weights"] = results_weighted
    
    # 3. Undersampling
    print(f"\n*** 3. RANDOM UNDERSAMPLING ***")
    X_under, y_under = create_balanced_sample(X_train, y_train, balance_method='undersample')
    analyze_class_distribution(y_under, "Undersampled Training Set")
    models_under = train_models_with_class_weights(X_under, y_under, use_class_weights=False)
    results_under = evaluate_with_threshold_tuning(models_under, X_val, X_test, y_val, y_test, "Undersampling")
    all_results["Undersampling"] = results_under
    
    # 4. Oversampling
    print(f"\n*** 4. RANDOM OVERSAMPLING ***")
    X_over, y_over = create_balanced_sample(X_train, y_train, balance_method='oversample')
    analyze_class_distribution(y_over, "Oversampled Training Set")
    models_over = train_models_with_class_weights(X_over, y_over, use_class_weights=False)
    results_over = evaluate_with_threshold_tuning(models_over, X_val, X_test, y_val, y_test, "Oversampling")
    all_results["Oversampling"] = results_over
    
    # Summary comparison
    print(f"\n{'='*90}")
    print("RECALL IMPROVEMENT SUMMARY")
    print("="*90)
    print(f"{'Technique':<20} | {'Best Model':<20} | {'Default Recall':<15} | {'Optimized Recall':<15} | {'Improvement':<12}")
    print("-"*90)
    
    for technique_name, results in all_results.items():
        best_default = max(results, key=lambda x: x['Test_Recall_Default'])
        best_optimal = max(results, key=lambda x: x['Test_Recall_Optimal'])
        improvement = best_optimal['Test_Recall_Optimal'] - best_default['Test_Recall_Default']
        
        print(f"{technique_name:<20} | {best_optimal['Model']:<20} | "
              f"{best_default['Test_Recall_Default']:<15.4f} | {best_optimal['Test_Recall_Optimal']:<15.4f} | "
              f"{improvement:<12.4f}")
    
    print("="*90)
    print("\nKEY INSIGHTS:")
    print("1. Class weights provide significant recall improvement with minimal computational cost")
    print("2. Threshold optimization can further boost recall but may reduce precision")
    print("3. Undersampling reduces training time but may lose important information")
    print("4. Oversampling can improve recall but increases training time")
    print("5. Monitor the precision-recall trade-off based on your business needs")
    
    # Find the best overall technique
    best_technique = max(all_results.items(), 
                        key=lambda x: max(result['Test_Recall_Optimal'] for result in x[1]))
    best_result = max(best_technique[1], key=lambda x: x['Test_Recall_Optimal'])
    
    print(f"\nRECOMMENDED APPROACH:")
    print(f"Technique: {best_technique[0]}")
    print(f"Model: {best_result['Model']}")
    print(f"Optimal Threshold: {best_result['Optimal_Threshold']:.3f}")
    print(f"Test Recall: {best_result['Test_Recall_Optimal']:.4f}")
    print(f"Test Precision: {best_result['Test_Precision_Optimal']:.4f}")
    print(f"Test F1-Score: {best_result['Test_F1_Optimal']:.4f}")

if __name__ == '__main__':
    main() 