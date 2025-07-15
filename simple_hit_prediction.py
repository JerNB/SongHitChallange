import csv
import random
import math

NUMERIC_COLS = [
    'energy','tempo','danceability','loudness','liveness','valence',
    'speechiness','instrumentalness','mode','key','duration_ms','acousticness'
]

DATA_PATH = 'song_data.csv'


def load_data(path=DATA_PATH, threshold=80):
    X, y = [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [float(row[col]) for col in NUMERIC_COLS]
                label = 1 if float(row['track_popularity']) >= threshold else 0
                X.append(features)
                y.append(label)
            except ValueError:
                # skip rows with missing values
                continue
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    return X, y


def standardize(X):
    n = len(X)
    m = len(X[0])
    means = [sum(row[j] for row in X) / n for j in range(m)]
    stds = [
        math.sqrt(sum((row[j] - means[j]) ** 2 for row in X) / n)
        for j in range(m)
    ]
    X_std = [
        [ (row[j] - means[j]) / stds[j] if stds[j] else 0.0 for j in range(m) ]
        for row in X
    ]
    return X_std, means, stds


def train_logistic(X, y, lr=0.01, epochs=200):
    n = len(X)
    m = len(X[0])
    w = [0.0] * m
    b = 0.0
    for _ in range(epochs):
        for i in range(n):
            z = sum(w[j] * X[i][j] for j in range(m)) + b
            pred = 1 / (1 + math.exp(-z))
            error = pred - y[i]
            for j in range(m):
                w[j] -= lr * error * X[i][j]
            b -= lr * error
    return w, b


def predict(X, w, b):
    preds = []
    for x in X:
        z = sum(w_j * x_j for w_j, x_j in zip(w, x)) + b
        prob = 1 / (1 + math.exp(-z))
        preds.append(1 if prob >= 0.5 else 0)
    return preds


def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def main():
    X, y = load_data()
    X_std, _, _ = standardize(X)
    split = int(0.8 * len(X_std))
    X_train, y_train = X_std[:split], y[:split]
    X_test, y_test = X_std[split:], y[split:]
    w, b = train_logistic(X_train, y_train)
    preds = predict(X_test, w, b)
    print('Accuracy:', accuracy(y_test, preds))


if __name__ == '__main__':
    main()
