# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

import whisper
import torch
from pyannote.audio import Pipeline

def download_model():
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=True)
    model = whisper.load_model("medium")

if __name__ == "__main__":
    download_model()
