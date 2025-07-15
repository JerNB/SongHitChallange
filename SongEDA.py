import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy.spatial import distance # use for question 3

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

df = pd.read_csv('SpotifyTrackDataset.csv',index_col=0)

# How many rows and columns?
nrows, ncols = df.shape
print(f'Dataset has {nrows} rows and {ncols} columns')

duplicated_rows = df.duplicated().sum()

# Are there any duplicated rows?
# if duplicated_rows == 0:
#     print('There are 0 rows that are duplicated, which means each row in the DataFrame is unique.')
#     print('So that we do not need to continue processing duplicate lines')
# else:
#     print(f'There are {duplicated_rows} rows that are duplicated so we need to drop those {duplicated_rows} rows')
# df = df.drop_duplicates()
# print(f'After drop duplicated rows, there are {df.shape[0]} rows left')

df.dtypes.to_frame('Data Type')

numerical_cols = df[df.columns[(df.dtypes == 'float64') | (df.dtypes == 'int64')]]
#print(numerical_cols.shape)

dist_numerical_cols = numerical_cols.describe().T[['min', 'max']]
dist_numerical_cols['Missing Values'] = numerical_cols.isnull().sum()
dist_numerical_cols['Missing Percentage'] = (numerical_cols.isnull().mean() * 100).round(2)
# The number of -1 values in the 'key' column
dist_numerical_cols.loc['key', 'Missing Values'] = (df['key'] == -1).sum()
#print(dist_numerical_cols.describe())

# 1. Find the index of the minimum loudness
max_idx = df['popularity'].idxmax()
# 2. Pull out that row
quietest = df.loc[max_idx, ['album_name', 'artists', 'popularity']]
print(quietest)

# 3. Plot the distribution of the 'popularity' column
# sns.set_style('darkgrid')
# sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
# numerical_cols.hist(figsize=(20,15), bins=30, xlabelsize=8, ylabelsize=8)
# plt.tight_layout()
# #plt.show()

categorical_cols = df[df.columns[(df.dtypes == 'object') | (df.dtypes == 'bool')]]
#print(categorical_cols.info())
dist_categorical_cols = pd.DataFrame(
    data = {
        'Missing Values': categorical_cols.isnull().sum(),
        'Missing Percentage': (categorical_cols.isnull().mean() * 100)
    }
)

categorical_cols[categorical_cols.isnull().any(axis=1)]

index_to_drop = df[categorical_cols.isnull().any(axis=1)].index
df.drop(index_to_drop, inplace=True)

# print(f'Rows with missing values dropped. Updated DataFrame shape: {df.shape}')

#print(df.describe(include=['object','bool']))

# Plotting the pie chart for the 'explicit' column
# unique_values, value_counts = np.unique(categorical_cols['explicit'], return_counts=True)
# fig, ax = plt.subplots(figsize=(5, 5))
# # Explode the slice with explicit tracks for emphasis
# explode = [0, 0.1]  # Only "yes" (true) will be slightly exploded
# colors = ['#66b3ff','#99ff99']
# ax.pie(value_counts, labels=unique_values, autopct='%1.2f%%', startangle=90, colors=colors， explode=explode)
# ax.axis('equal')
# ax.set_title('Distribution of Explicit Tracks')
# plt.show()


# Plotting the distribution of top10 categorical columns
top_n = 10
sns.set_style('darkgrid')
sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
# Get the top N most frequent artists, albums, tracks, and genres
top_artists = df['artists'].value_counts().head(top_n)
top_albums = df['album_name'].value_counts().head(top_n)
top_tracks = df['track_name'].value_counts().head(top_n)
top_genres = df['track_genre'].value_counts().head(top_n)

# Finding the top 10 artists, albums, tracks, and genres
# # Disable FutureWarning
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=FutureWarning)

#     # Plotting
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

#     # Top N Artists
#     sns.barplot(x=top_artists.values, y=top_artists.index, palette="crest", ax=axes[0, 0], orient='h',  zorder=3, width=0.5)
#     axes[0, 0].set_title(f'Top {top_n} Artists')
#     axes[0, 0].set_xlabel('Frequency')
#     axes[0, 0].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

#         # Top N Albums
#     sns.barplot(x=top_albums.values, y=top_albums.index, palette="crest", ax=axes[0, 1], orient='h', zorder=3, width=0.5)
#     axes[0, 1].set_title(f'Top {top_n} Albums')
#     axes[0, 1].set_xlabel('Frequency')
#     axes[0, 1].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

#     # Top N Tracks
#     sns.barplot(x=top_tracks.values, y=top_tracks.index, palette="crest", ax=axes[1, 0], orient='h', zorder=3, width=0.5)
#     axes[1, 0].set_title(f'Top {top_n} Tracks')
#     axes[1, 0].set_xlabel('Frequency')
#     axes[1, 0].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

#     # Top N Genres
#     sns.barplot(x=top_genres.values, y=top_genres.index, palette="crest", ax=axes[1, 1], orient='h', zorder=3, width=0.5)
#     axes[1, 1].set_title(f'Top {top_n} Genres')
#     axes[1, 1].set_xlabel('Frequency')
#     axes[1, 1].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

#     plt.tight_layout()
#     plt.show()

# plotting the abnormality of numerical columns
# boxplot for numerical columns
sns.set_style('darkgrid')
sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
columns = ['popularity', 'duration_ms', 'tempo', 'loudness', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
for i, col in enumerate(columns):
    sns.boxplot(y=col, data=numerical_cols, ax=axes[i//4, i%4])
    axes[i//4, i%4].set_title(col)
plt.tight_layout()
plt.show()

