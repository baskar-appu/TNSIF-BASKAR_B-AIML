import gradio as gr
import pandas as pd
import joblib

from fastapi import FastAPI
from gradio.routes import mount_gradio_app


# ============================================================
# LOAD TRAINED MODEL AND SCALER
# ============================================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# ============================================================
# DETERMINE LISTENER SEGMENT
# ============================================================

def get_segment(cluster):

    # Convert cluster centres back to original scale
    centres = scaler.inverse_transform(
        model.cluster_centers_
    )

    activity_scores = []

    for centre in centres:

        listening_hours = centre[0]
        songs_per_day = centre[1]
        skip_rate = centre[2]
        playlist_count = centre[3]

        # Higher score = higher listening activity
        score = (
            listening_hours
            + songs_per_day
            + playlist_count
            - skip_rate
        )

        activity_scores.append(score)

    # Sort clusters from lowest to highest activity
    sorted_clusters = sorted(
        range(len(activity_scores)),
        key=lambda x: activity_scores[x]
    )

    casual_cluster = sorted_clusters[0]
    explorer_cluster = sorted_clusters[1]
    heavy_cluster = sorted_clusters[2]

    if cluster == casual_cluster:
        return "🎧 Casual Listener"

    elif cluster == explorer_cluster:
        return "🎵 Music Explorer"

    elif cluster == heavy_cluster:
        return "🔥 Heavy Listener"

    return "Unknown"


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_listener(
    listening_hours,
    songs_per_day,
    skip_rate,
    playlist_count
):

    # --------------------------------------------------------
    # Create Pandas DataFrame
    # --------------------------------------------------------
    # This fixes the StandardScaler feature-name warning.
    # The column names and order are exactly the same
    # as those used during training.
    # --------------------------------------------------------

    user_data = pd.DataFrame([{

        "listening_hours_per_week":
            listening_hours,

        "songs_per_day":
            songs_per_day,

        "skip_rate":
            skip_rate,

        "playlist_count":
            playlist_count

    }])


    # --------------------------------------------------------
    # Scale using the SAME scaler
    # --------------------------------------------------------

    user_data_scaled = scaler.transform(
        user_data
    )


    # --------------------------------------------------------
    # Predict cluster
    # --------------------------------------------------------

    cluster = int(
        model.predict(
            user_data_scaled
        )[0]
    )


    # --------------------------------------------------------
    # Convert cluster into meaningful segment
    # --------------------------------------------------------

    segment = get_segment(cluster)


    # --------------------------------------------------------
    # Get cluster centre
    # --------------------------------------------------------

    centres = scaler.inverse_transform(
        model.cluster_centers_
    )

    centre = centres[cluster]


    # --------------------------------------------------------
    # Create result
    # --------------------------------------------------------

    result = f"""
### {segment}

**K-Means Cluster:** {cluster}

---

### Listener Information

| Feature | Value |
|---|---:|
| Listening Hours/Week | {listening_hours:.1f} |
| Songs/Day | {songs_per_day:.1f} |
| Skip Rate | {skip_rate:.1f}% |
| Playlist Count | {playlist_count} |

---

### Cluster Centre

| Feature | Cluster Average |
|---|---:|
| Listening Hours/Week | {centre[0]:.2f} |
| Songs/Day | {centre[1]:.2f} |
| Skip Rate | {centre[2]:.2f}% |
| Playlist Count | {centre[3]:.2f} |

"""

    return result


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Music Listener Segmentation API",
    description="K-Means based music listener segmentation",
    version="1.0"
)


# ============================================================
# PYTHON-BASED INTERACTIVE UI
# ============================================================

with gr.Blocks(
    title="Music Listener Segmentation"
) as demo:

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    gr.Markdown(
        """
# 🎵 Music Listener Segmentation

### Unsupervised Machine Learning using K-Means

Enter the listener's behaviour using the interactive
controls below to identify their listener segment.
"""
    )


    # --------------------------------------------------------
    # Input section
    # --------------------------------------------------------

    gr.Markdown(
        "## 🎧 Listener Behaviour"
    )


    with gr.Row():

        listening_hours = gr.Slider(
            minimum=0,
            maximum=168,
            value=10,
            step=0.5,
            label="Listening Hours per Week"
        )

        songs_per_day = gr.Slider(
            minimum=0,
            maximum=150,
            value=20,
            step=1,
            label="Songs per Day"
        )


    with gr.Row():

        skip_rate = gr.Slider(
            minimum=0,
            maximum=100,
            value=25,
            step=1,
            label="Skip Rate (%)"
        )

        playlist_count = gr.Slider(
            minimum=0,
            maximum=100,
            value=5,
            step=1,
            label="Playlist Count"
        )


    # --------------------------------------------------------
    # Analyze button
    # --------------------------------------------------------

    analyze_button = gr.Button(
        "🔍 Analyze Listener",
        variant="primary"
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = gr.Markdown(
        value="""
### 👈 Enter listener information

Move the sliders and click **Analyze Listener**.
"""
    )


    # --------------------------------------------------------
    # Button action
    # --------------------------------------------------------

    analyze_button.click(

        fn=predict_listener,

        inputs=[
            listening_hours,
            songs_per_day,
            skip_rate,
            playlist_count
        ],

        outputs=result
    )


    # --------------------------------------------------------
    # Information section
    # --------------------------------------------------------

    gr.Markdown(
        """
---

## 📊 Features Used

The K-Means model uses four behavioural features:

- **Listening Hours per Week**
- **Songs per Day**
- **Skip Rate**
- **Playlist Count**

### Clustering

The model automatically groups listeners into:

- 🎧 Casual Listener
- 🎵 Music Explorer
- 🔥 Heavy Listener

The labels are assigned **after analysing the K-Means
cluster centres**. K-Means itself only produces cluster
numbers.
"""
    )


# ============================================================
# MOUNT GRADIO UI INTO FASTAPI
# ============================================================

app = mount_gradio_app(
    app,
    demo,
    path="/"
)