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
ncols, nrows = df.shape
print(f'Dataset has {ncols} rows and {nrows} columns')

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
sns.set_style('darkgrid')
sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
numerical_cols.hist(figsize=(20,15), bins=30, xlabelsize=8, ylabelsize=8)
plt.tight_layout()
plt.show()

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

print(f'Rows with missing values dropped. Updated DataFrame shape: {df.shape}')

print(df.describe(include=['object','bool']))

#Plotting the pie chart for the 'explicit' column
unique_values, value_counts = np.unique(categorical_cols['explicit'], return_counts=True)
fig, ax = plt.subplots(figsize=(5, 5))
# Explode the slice with explicit tracks for emphasis
explode = [0, 0.1]  # Only "yes" (true) will be slightly exploded
colors = ['#66b3ff','#99ff99']
ax.pie(value_counts, labels=unique_values, autopct='%1.2f%%', startangle=90, colors=colors, explode=explode)
ax.axis('equal')
ax.set_title('Distribution of Explicit Tracks')
plt.show()


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
# Disable FutureWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    # Top N Artists
    sns.barplot(x=top_artists.values, y=top_artists.index, palette="crest", ax=axes[0, 0], orient='h',  zorder=3, width=0.5)
    axes[0, 0].set_title(f'Top {top_n} Artists')
    axes[0, 0].set_xlabel('Frequency')
    axes[0, 0].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

        # Top N Albums
    sns.barplot(x=top_albums.values, y=top_albums.index, palette="crest", ax=axes[0, 1], orient='h', zorder=3, width=0.5)
    axes[0, 1].set_title(f'Top {top_n} Albums')
    axes[0, 1].set_xlabel('Frequency')
    axes[0, 1].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

    # Top N Tracks
    sns.barplot(x=top_tracks.values, y=top_tracks.index, palette="crest", ax=axes[1, 0], orient='h', zorder=3, width=0.5)
    axes[1, 0].set_title(f'Top {top_n} Tracks')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

    # Top N Genres
    sns.barplot(x=top_genres.values, y=top_genres.index, palette="crest", ax=axes[1, 1], orient='h', zorder=3, width=0.5)
    axes[1, 1].set_title(f'Top {top_n} Genres')
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

    plt.tight_layout()
    plt.show()

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

# heatmap for correlation
corr = numerical_cols.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.set_style('white')
sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
plt.figure(figsize=(15, 10))
sns.heatmap(corr, mask=mask, annot=True, vmin=-1, vmax=1,cmap='coolwarm')
plt.show()

avg_popularity_by_genre = df.groupby('track_genre')['popularity'].mean().reset_index()
top10_popular_genres = avg_popularity_by_genre.nlargest(10, 'popularity')

plt.figure(figsize=(10, 6))
sns.barplot(x='popularity', y='track_genre', data=top10_popular_genres)
plt.title('Top 10 Genres with Highest Average Popularity')
plt.xlabel('Average popularity score')
plt.ylabel('Genre')
plt.show()

# =============================================================================
# ADDITIONAL ANALYSIS FOR ESSAY - HIT PREDICTION INSIGHTS
# =============================================================================

# 1. Create binary hit classification (top 20% as hits)
hit_threshold = np.percentile(df['popularity'], 80)
df['is_hit'] = (df['popularity'] >= hit_threshold).astype(int)

print(f"\n{'='*60}")
print("HIT CLASSIFICATION ANALYSIS")
print("="*60)
print(f"Hit threshold (80th percentile): {hit_threshold:.1f}")
print(f"Number of hits: {df['is_hit'].sum():,} ({df['is_hit'].mean()*100:.1f}%)")
print(f"Number of non-hits: {(1-df['is_hit']).sum():,} ({(1-df['is_hit']).mean()*100:.1f}%)")

# 2. Audio Feature Comparison: Hits vs Non-Hits
audio_features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence',
                 'speechiness', 'instrumentalness', 'acousticness', 'duration_ms']

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
fig.suptitle('Audio Features: Hits vs Non-Hits Distribution', fontsize=16, fontweight='bold')

for i, feature in enumerate(audio_features):
    row, col = i // 4, i % 4
    
    # Box plot comparing hits vs non-hits
    sns.boxplot(x='is_hit', y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(f'{feature.title()}')
    axes[row, col].set_xlabel('Hit (1) vs Non-Hit (0)')
    
    # Calculate and display mean difference
    hit_mean = df[df['is_hit']==1][feature].mean()
    non_hit_mean = df[df['is_hit']==0][feature].mean()
    diff_pct = ((hit_mean - non_hit_mean) / non_hit_mean) * 100
    
    axes[row, col].text(0.5, 0.95, f'Δ: {diff_pct:+.1f}%', 
                       transform=axes[row, col].transAxes, 
                       ha='center', va='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Remove empty subplot
axes[2, 3].remove()
plt.tight_layout()
plt.show()

# 3. Feature Correlation with Popularity
correlations = df[audio_features + ['popularity']].corr()['popularity'].drop('popularity').sort_values(key=abs, ascending=False)

plt.figure(figsize=(12, 8))
colors = ['red' if x < 0 else 'green' for x in correlations.values]
bars = plt.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
plt.yticks(range(len(correlations)), correlations.index)
plt.xlabel('Correlation with Popularity')
plt.title('Audio Features Correlation with Song Popularity', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Add correlation values on bars
for i, (bar, val) in enumerate(zip(bars, correlations.values)):
    plt.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}', 
             va='center', ha='left' if val > 0 else 'right', fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Genre Performance Analysis
genre_analysis = df.groupby('track_genre').agg({
    'popularity': ['mean', 'std', 'count'],
    'is_hit': ['sum', 'mean']
}).round(3)

genre_analysis.columns = ['avg_popularity', 'std_popularity', 'song_count', 'hit_count', 'hit_rate']
genre_analysis = genre_analysis[genre_analysis['song_count'] >= 10]  # Only genres with 10+ songs
genre_analysis = genre_analysis.sort_values('hit_rate', ascending=False)

# Plot hit rate by genre
plt.figure(figsize=(14, 8))
top_15_genres = genre_analysis.head(15)
bars = plt.barh(range(len(top_15_genres)), top_15_genres['hit_rate'], 
                color=plt.cm.viridis(top_15_genres['hit_rate']))
plt.yticks(range(len(top_15_genres)), top_15_genres.index)
plt.xlabel('Hit Rate (Proportion of Top 20% Popular Songs)')
plt.title('Hit Rate by Genre (Top 15 Genres)', fontsize=14, fontweight='bold')

# Add hit rate values on bars
for i, (bar, val) in enumerate(zip(bars, top_15_genres['hit_rate'])):
    plt.text(val + 0.01, i, f'{val:.3f}', va='center', ha='left', fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Audio Feature Radar Chart - Hits vs Non-Hits Profile
# Normalize features for radar chart
features_for_radar = ['energy', 'danceability', 'loudness', 'liveness', 'valence', 'acousticness']
df_normalized = df.copy()

# Min-max normalization for radar chart
for feature in features_for_radar:
    min_val = df[feature].min()
    max_val = df[feature].max()
    df_normalized[feature] = (df[feature] - min_val) / (max_val - min_val)

# Calculate mean profiles
hit_profile = df_normalized[df_normalized['is_hit']==1][features_for_radar].mean()
non_hit_profile = df_normalized[df_normalized['is_hit']==0][features_for_radar].mean()

# Create radar chart
angles = np.linspace(0, 2*np.pi, len(features_for_radar), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

hit_values = hit_profile.values.tolist()
hit_values += hit_values[:1]

non_hit_values = non_hit_profile.values.tolist()
non_hit_values += non_hit_values[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
ax.plot(angles, hit_values, 'o-', linewidth=2, label='Hits', color='red', alpha=0.7)
ax.fill(angles, hit_values, alpha=0.2, color='red')
ax.plot(angles, non_hit_values, 'o-', linewidth=2, label='Non-Hits', color='blue', alpha=0.7)
ax.fill(angles, non_hit_values, alpha=0.2, color='blue')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features_for_radar)
ax.set_ylim(0, 1)
ax.set_title('Audio Feature Profile: Hits vs Non-Hits', size=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.tight_layout()
plt.show()

# 6. Statistical Summary for Essay
print(f"\n{'='*60}")
print("KEY INSIGHTS FOR ESSAY")
print("="*60)

# Most discriminative features
top_3_features = correlations.abs().nlargest(3)
print(f"Most predictive audio features for popularity:")
for i, (feature, corr) in enumerate(top_3_features.items(), 1):
    direction = "positively" if corr > 0 else "negatively"
    print(f"  {i}. {feature.title()} (r={corr:.3f}) - {direction} correlated")

# Genre insights
best_genre = genre_analysis.index[0]
worst_genre = genre_analysis.index[-1]
print(f"\nGenre with highest hit rate: {best_genre} ({genre_analysis.loc[best_genre, 'hit_rate']:.3f})")
print(f"Genre with lowest hit rate: {worst_genre} ({genre_analysis.loc[worst_genre, 'hit_rate']:.3f})")

# Feature differences
print(f"\nKey differences between hits and non-hits:")
for feature in audio_features[:5]:  # Top 5 features
    hit_mean = df[df['is_hit']==1][feature].mean()
    non_hit_mean = df[df['is_hit']==0][feature].mean()
    diff_pct = ((hit_mean - non_hit_mean) / non_hit_mean) * 100
    direction = "higher" if diff_pct > 0 else "lower"
    print(f"  • {feature.title()}: Hits have {abs(diff_pct):.1f}% {direction} values on average")

print("="*60)