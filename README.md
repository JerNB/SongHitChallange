# Hit Song Prediction

This repository contains datasets of Spotify tracks and a script to train various models for predicting whether a song will become a "hit" based on audio features.

## Data
- `song_data.csv`: Spotify audio features and metadata.
- `SpotifyTrackDataset.csv`: Additional dataset used for EDA in `SongEDA.py`.

A song is labeled as a hit if its `track_popularity` is 80 or greater. You can adjust this threshold in `song_hit_prediction.py`.

## Usage
Run the following command to train and evaluate all models:

```bash
python song_hit_prediction.py
```

The script trains logistic regression, support vector machine, random forest, and a neural network using scikit-learn. Model accuracies and classification reports are printed to the console.
