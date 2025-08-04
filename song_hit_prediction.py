import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Try to import XGBoost, handle if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False
# Commenting out imblearn imports that are causing issues
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.combine import SMOTEENN
from collections import Counter

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

def load_data(path=DATA_PATH, hit_percentile=80):
    """Load the Spotify song dataset and create binary label 'hit' based on popularity percentile."""
    df = pd.read_csv(path)
    
    # Calculate the threshold based on percentile (e.g., top 20% = 80th percentile)
    popularity_threshold = np.percentile(df['track_popularity'], hit_percentile)
    df['hit'] = (df['track_popularity'] >= popularity_threshold).astype(int)
    
    print(f"Using {100 - hit_percentile}% of songs as hits (popularity >= {popularity_threshold:.1f})")
    
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
    
    # Split into train+validation (80%) and test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nAfter train/test split:")
    analyze_class_distribution(y_temp, "Training+Validation Set (for CV)")
    analyze_class_distribution(y_test, "Test Set")
    
    return X_temp, X_test, y_temp, y_test

def simple_undersample(X_train, y_train):
    """Simple undersampling using built-in numpy methods."""
    counter = Counter(y_train)
    majority_class = max(counter, key=counter.get)
    minority_class = min(counter, key=counter.get)
    
    majority_indices = np.where(y_train == majority_class)[0]
    minority_indices = np.where(y_train == minority_class)[0]
    
    # Random undersample majority class
    np.random.seed(42)
    selected_majority = np.random.choice(majority_indices, size=len(minority_indices), replace=False)
    balanced_indices = np.concatenate([selected_majority, minority_indices])
    
    # Shuffle the indices
    np.random.seed(42)
    np.random.shuffle(balanced_indices)
    
    X_balanced = X_train.iloc[balanced_indices]
    y_balanced = y_train.iloc[balanced_indices]
    
    return X_balanced, y_balanced

def create_models():
    """Create model pipelines for cross-validation."""
    models = {}

    # Logistic Regression with balanced class weights
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    models['Logistic Regression'] = lr_pipe

    # Support Vector Machine with balanced class weights
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
    ])
    models['SVM'] = svm_pipe

    # Random Forest with balanced class weights
    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
    ])
    models['Random Forest'] = rf_pipe

    # Neural Network (MLP)
    mlp_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42))
    ])
    models['Neural Network'] = mlp_pipe

    # XGBoost with balanced class weights (only if available)
    if XGBOOST_AVAILABLE:
        xgb_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=4  # Handle class imbalance (approximate ratio)
            ))
        ])
        models['XGBoost'] = xgb_pipe
    else:
        print("Skipping XGBoost - not available")

    return models

def perform_cross_validation(models, X, y, cv_folds=5):
    """Perform k-fold cross-validation on all models."""
    print(f"\n{'='*90}")
    print(f"K-FOLD CROSS-VALIDATION RESULTS (k={cv_folds})")
    print("="*90)
    
    # Create stratified k-fold to maintain class distribution
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Define scoring metrics
    scorers = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0)
    }
    
    cv_results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        model_results = {}
        
        for metric_name, scorer in scorers.items():
            scores = cross_val_score(model, X, y, cv=skf, scoring=scorer, n_jobs=-1)
            model_results[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
        
        cv_results[model_name] = model_results
    
    return cv_results

def display_cv_results(cv_results):
    """Display cross-validation results in a formatted table."""
    print(f"\n{'='*100}")
    print(f"{'Model':<20} | {'Accuracy':<20} | {'Precision':<20} | {'Recall':<20} | {'F1-Score':<20}")
    print(f"{'':20} | {'Mean ± Std':<20} | {'Mean ± Std':<20} | {'Mean ± Std':<20} | {'Mean ± Std':<20}")
    print("-"*100)
    
    for model_name, results in cv_results.items():
        acc_mean, acc_std = results['accuracy']['mean'], results['accuracy']['std']
        prec_mean, prec_std = results['precision']['mean'], results['precision']['std']
        rec_mean, rec_std = results['recall']['mean'], results['recall']['std']
        f1_mean, f1_std = results['f1']['mean'], results['f1']['std']
        
        print(f"{model_name:<20} | "
              f"{acc_mean:.3f} ± {acc_std:.3f}    | "
              f"{prec_mean:.3f} ± {prec_std:.3f}    | "
              f"{rec_mean:.3f} ± {rec_std:.3f}    | "
              f"{f1_mean:.3f} ± {f1_std:.3f}    ")
    
    print("="*100)

def train_models_with_balancing(X_train, y_train):
    """Train multiple classifiers with class weight balancing."""
    models = {}

    # Logistic Regression with balanced class weights
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    lr_pipe.fit(X_train, y_train)
    models['Logistic Regression'] = lr_pipe

    # Support Vector Machine with balanced class weights
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
    ])
    svm_pipe.fit(X_train, y_train)
    models['SVM'] = svm_pipe

    # Random Forest with balanced class weights
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_clf.fit(X_train, y_train)
    models['Random Forest'] = rf_clf

    # Neural Network (MLP) - doesn't support class_weight directly
    mlp_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42))
    ])
    mlp_pipe.fit(X_train, y_train)
    models['Neural Network'] = mlp_pipe

    # XGBoost with balanced class weights (only if available)
    if XGBOOST_AVAILABLE:
        xgb_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=4  # Handle class imbalance (approximate ratio)
            ))
        ])
        xgb_pipe.fit(X_train, y_train)
        models['XGBoost'] = xgb_pipe
    else:
        print("Skipping XGBoost - not available")

    return models

def evaluate_on_test_set(models, X_test, y_test):
    """Evaluate trained models on the test set."""
    print(f"\n{'='*90}")
    print("FINAL TEST SET EVALUATION")
    print("="*90)
    
    results = []
    
    for name, model in models.items():
        test_preds = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds, zero_division=0)
        test_recall = recall_score(y_test, test_preds, zero_division=0)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        
        results.append({
            'Model': name,
            'Test_Accuracy': test_accuracy,
            'Test_Precision': test_precision,
            'Test_Recall': test_recall,
            'Test_F1': test_f1
        })
    
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-"*70)
    
    for result in results:
        print(f"{result['Model']:<20} | "
              f"{result['Test_Accuracy']:<10.4f} | "
              f"{result['Test_Precision']:<10.4f} | "
              f"{result['Test_Recall']:<10.4f} | "
              f"{result['Test_F1']:<10.4f}")
    
    print("="*90)
    return results

def main():
    # Load and analyze data
    X_train_val, X_test, y_train_val, y_test = load_data()
    
    print("\n" + "="*90)
    print("MODEL EVALUATION WITH K-FOLD CROSS-VALIDATION")
    print("="*90)
    
    # Create models for cross-validation
    models = create_models()
    
    # Perform cross-validation on training+validation data
    print(f"\n*** Performing 5-Fold Cross-Validation ***")
    cv_results = perform_cross_validation(models, X_train_val, y_train_val, cv_folds=5)
    display_cv_results(cv_results)
    
    # Train final models on full training+validation data and evaluate on test set
    print(f"\n*** Training Final Models on Full Training Data ***")
    final_models = train_models_with_balancing(X_train_val, y_train_val)
    test_results = evaluate_on_test_set(final_models, X_test, y_test)
    
    # Test with undersampling and cross-validation
    print(f"\n*** Cross-Validation with Undersampling ***")
    X_under, y_under = simple_undersample(X_train_val, y_train_val)
    analyze_class_distribution(y_under, "Undersampled Training Set")
    
    cv_results_under = perform_cross_validation(models, X_under, y_under, cv_folds=5)
    display_cv_results(cv_results_under)
    
    # Final recommendations
    print(f"\n{'='*90}")
    print("CROSS-VALIDATION SUMMARY AND RECOMMENDATIONS")
    print("="*90)
    
    # Find best performing model from CV
    best_model_name = None
    best_f1_score = 0
    
    for model_name, results in cv_results.items():
        f1_mean = results['f1']['mean']
        if f1_mean > best_f1_score:
            best_f1_score = f1_mean
            best_model_name = model_name
    
    print(f"Best performing model (CV F1-score): {best_model_name} ({best_f1_score:.3f})")
    print("\nKey Insights:")
    print("1. Cross-validation provides more robust performance estimates")
    print("2. Standard deviation indicates model stability across folds")
    print("3. Compare CV results with final test performance to check for overfitting")
    print("4. Low std values indicate consistent performance across different data splits")

if __name__ == '__main__':
    main()
