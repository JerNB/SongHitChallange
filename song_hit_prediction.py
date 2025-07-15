import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Load data
DATA_PATH = 'song_data.csv'

def load_data(path=DATA_PATH, popularity_threshold=80):
    """Load the Spotify song dataset and create binary label 'hit' based on popularity."""
    df = pd.read_csv(path)
    df['hit'] = (df['track_popularity'] >= popularity_threshold).astype(int)
    numeric_cols = [
        'energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence',
        'speechiness', 'instrumentalness', 'mode', 'key', 'duration_ms', 'acousticness'
    ]
    X = df[numeric_cols]
    y = df['hit']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_models(X_train, y_train):
    """Train multiple classifiers and return fitted models."""
    models = {}

    # Logistic Regression
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    lr_pipe.fit(X_train, y_train)
    models['LogisticRegression'] = lr_pipe

    # Support Vector Machine
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True))
    ])
    svm_pipe.fit(X_train, y_train)
    models['SVM'] = svm_pipe

    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_clf.fit(X_train, y_train)
    models['RandomForest'] = rf_clf

    # Neural Network (MLP)
    mlp_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42))
    ])
    mlp_pipe.fit(X_train, y_train)
    models['NeuralNetwork'] = mlp_pipe

    return models


def evaluate_models(models, X_test, y_test):
    """Print accuracy for each model."""
    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))


def main():
    X_train, X_test, y_train, y_test = load_data()
    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)

if __name__ == '__main__':
    main()
