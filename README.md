# Hit Song Prediction

This repository contains datasets of Spotify tracks and scripts to predict whether a song will become a "hit" based on audio features.

## Data
- `song_data.csv`: Spotify audio features and metadata.
- `SpotifyTrackDataset.csv`: Additional dataset used for EDA in `SongEDA.py`.

A song is labeled as a hit if its `track_popularity` is 80 or greater. You can adjust this threshold in the scripts.

## Usage
Two prediction scripts are provided:

1. **`song_hit_prediction.py`** – uses `pandas` and `scikit-learn` to train logistic regression, SVM, random forest, and a neural network. Run it with:

```bash
python song_hit_prediction.py
```

Model accuracies and classification reports will be printed.

2. **`simple_hit_prediction.py`** – a pure Python implementation of logistic regression that does not require external packages. Use this script when you cannot install dependencies. Run it with:

```bash
python simple_hit_prediction.py
```

The script outputs the accuracy of a basic logistic regression model.
