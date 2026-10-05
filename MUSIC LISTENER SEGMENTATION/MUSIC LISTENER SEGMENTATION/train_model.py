import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

data = pd.read_csv("music_listeners.csv")

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(data.shape)


# --------------------------------------------------
# Select features
# --------------------------------------------------

features = [
    "listening_hours_per_week",
    "songs_per_day",
    "skip_rate",
    "playlist_count"
]

X = data[features]


# --------------------------------------------------
# Scale features
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# Create K-Means
# --------------------------------------------------

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# --------------------------------------------------
# Train
# --------------------------------------------------

model.fit(X_scaled)


# --------------------------------------------------
# Display cluster centres
# --------------------------------------------------

centres = scaler.inverse_transform(
    model.cluster_centers_
)


print("\nCluster Centres:")

for i, centre in enumerate(centres):

    print(f"\nCluster {i}")

    print(
        f"Listening Hours/Week: {centre[0]:.2f}"
    )

    print(
        f"Songs/Day: {centre[1]:.2f}"
    )

    print(
        f"Skip Rate: {centre[2]:.2f}%"
    )

    print(
        f"Playlist Count: {centre[3]:.2f}"
    )


# --------------------------------------------------
# Save model and scaler
# --------------------------------------------------

joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)


print("\nModel saved: model.pkl")
print("Scaler saved: scaler.pkl")
print("\nTraining completed successfully!")