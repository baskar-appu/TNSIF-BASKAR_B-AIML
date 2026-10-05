import pandas as pd
import numpy as np


# Make results reproducible
np.random.seed(42)


# Number of listeners
n_listeners = 300


# --------------------------------------------------
# Generate different listener behaviour patterns
# --------------------------------------------------

# 1. Casual listeners
casual_count = 100

casual_hours = np.random.normal(
    loc=5,
    scale=2,
    size=casual_count
)

casual_songs = np.random.normal(
    loc=10,
    scale=3,
    size=casual_count
)

casual_skip = np.random.normal(
    loc=45,
    scale=8,
    size=casual_count
)

casual_playlists = np.random.normal(
    loc=3,
    scale=1.5,
    size=casual_count
)


# 2. Music explorers
explorer_count = 100

explorer_hours = np.random.normal(
    loc=15,
    scale=3,
    size=explorer_count
)

explorer_songs = np.random.normal(
    loc=25,
    scale=5,
    size=explorer_count
)

explorer_skip = np.random.normal(
    loc=25,
    scale=6,
    size=explorer_count
)

explorer_playlists = np.random.normal(
    loc=9,
    scale=2,
    size=explorer_count
)


# 3. Heavy listeners
heavy_count = 100

heavy_hours = np.random.normal(
    loc=30,
    scale=5,
    size=heavy_count
)

heavy_songs = np.random.normal(
    loc=50,
    scale=8,
    size=heavy_count
)

heavy_skip = np.random.normal(
    loc=8,
    scale=3,
    size=heavy_count
)

heavy_playlists = np.random.normal(
    loc=20,
    scale=4,
    size=heavy_count
)


# --------------------------------------------------
# Combine all listeners
# --------------------------------------------------

listening_hours = np.concatenate([
    casual_hours,
    explorer_hours,
    heavy_hours
])

songs_per_day = np.concatenate([
    casual_songs,
    explorer_songs,
    heavy_songs
])

skip_rate = np.concatenate([
    casual_skip,
    explorer_skip,
    heavy_skip
])

playlist_count = np.concatenate([
    casual_playlists,
    explorer_playlists,
    heavy_playlists
])


# --------------------------------------------------
# Keep values realistic
# --------------------------------------------------

listening_hours = np.clip(
    listening_hours,
    0,
    168
)

songs_per_day = np.clip(
    songs_per_day,
    1,
    200
)

skip_rate = np.clip(
    skip_rate,
    0,
    100
)

playlist_count = np.clip(
    playlist_count,
    1,
    100
)


# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

data = pd.DataFrame({

    "listening_hours_per_week":
        np.round(listening_hours, 2),

    "songs_per_day":
        np.round(songs_per_day, 2),

    "skip_rate":
        np.round(skip_rate, 2),

    "playlist_count":
        np.round(playlist_count).astype(int)
})


# Shuffle rows
data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Save dataset
# --------------------------------------------------

data.to_csv(
    "music_listeners.csv",
    index=False
)


print("Dataset created successfully!")
print(f"Total listeners: {len(data)}")

print("\nFirst 10 records:")
print(data.head(10))

print("\nDataset saved as:")
print("music_listeners.csv")