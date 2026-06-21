# Music Genre Classification

A Google Colab-based machine learning project for classifying music genres with the GTZAN dataset and an additional Turkish Rap class. The notebooks cover data acquisition, audio preprocessing, feature extraction, model training, model comparison, visualization, and a small Gemini-powered multi-agent career assistant experiment.

## Project Goals

- Build an end-to-end audio classification workflow.
- Extend the standard GTZAN genre set with Turkish Rap samples.
- Compare handcrafted audio features with convolutional neural networks.
- Evaluate multiple classical machine learning algorithms.
- Analyze confusion between Hip-Hop and Turkish Rap.
- Save reusable features, models, plots, and experiment reports to Google Drive.

## Notebooks

### `music_genre_classification.ipynb`

The main experiment notebook includes:

- Kaggle dataset download and extraction
- Google Drive integration
- Optional Turkish Rap audio collection with `yt-dlp`
- Audio conversion through FFmpeg
- Fixed-duration audio preprocessing
- MFCC and Mel Spectrogram extraction
- Random Forest, SVM, KNN, Gradient Boosting, CNN, and Voting Classifier experiments
- Classification reports and confusion matrices
- Feature-set ablation experiments
- Model and result export

### `agentic_career_assistant.ipynb`

A separate Gemini API experiment that coordinates multiple role-focused agents to produce:

- Profile analysis
- Internship-role fit analysis
- CV-ready project descriptions
- A short application email
- Skill-gap recommendations
- A final reviewed report

## Audio Features

The experiments use or analyze:

- Mel-frequency cepstral coefficients (MFCC)
- Mel Spectrograms
- Chroma features
- Zero-Crossing Rate
- Spectral Centroid

The notebook stores extracted arrays as NumPy files so model experiments can be repeated without processing every audio file again.

## Models

The project compares:

- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Gradient Boosting
- Convolutional Neural Networks
- Soft-voting ensembles

The notebook records a Voting Classifier result of approximately 75% accuracy for one experiment. Results depend on the available audio files, preprocessing choices, random split, and runtime environment.

## Running in Google Colab

1. Open `music_genre_classification.ipynb` in Google Colab.
2. Mount Google Drive when prompted.
3. Create a Kaggle API token from your Kaggle account.
4. Upload `kaggle.json` only to the temporary Colab session.
5. Run the notebook cells in order.

The notebook uses paths under:

```text
/content/drive/MyDrive/music_genre_project/
```

You can change these paths if you prefer a different Drive directory.

## Gemini API Setup

Do not place an API key directly in a notebook. Add `GEMINI_API_KEY` to the Google Colab **Secrets** panel and load it at runtime:

```python
import os
from google.colab import userdata

os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```

Then open and run `agentic_career_assistant.ipynb`.

## Local Installation

The notebooks are designed primarily for Colab, but their Python dependencies can be installed locally:

```bash
git clone https://github.com/melekbspnr/MusicGenreClassification.git
cd MusicGenreClassification

python -m venv .venv
pip install -r requirements.txt
jupyter notebook
```

Colab-specific Drive, file-upload, and shell-command cells must be adapted for a local environment. FFmpeg must also be installed separately.

## Generated Artifacts

Depending on which cells are executed, the workflow may create:

- `X_mfcc.npy`
- `X_mel.npy`
- `y_labels.npy`
- Saved Keras models
- Pickled Voting Classifier models
- Confusion matrices
- Model-comparison charts
- Ablation-study results
- Text experiment summaries

These files are intentionally excluded from Git because they can be large or environment-specific.

## Repository Structure

```text
.
|-- music_genre_classification.ipynb
|-- agentic_career_assistant.ipynb
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Data and Security Notes

- GTZAN is downloaded from Kaggle and is not redistributed in this repository.
- Verify the usage rights of any additional audio collected from external platforms.
- `kaggle.json`, API keys, downloaded audio, model files, and generated datasets must not be committed.
- Notebook outputs were cleared before publication to avoid leaking local paths, credentials, or generated personal content.

## Limitations

- GTZAN contains known duplicates and possible labeling issues.
- The additional Turkish Rap data may not be balanced against every GTZAN class.
- Results can change with dataset composition and random splitting.
- The notebooks prioritize experimentation and documentation over a production inference pipeline.
