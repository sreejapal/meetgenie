from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os
import soundfile as sf
import torch

load_dotenv()

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=os.getenv("HF_TOKEN")
)

def diarize(audio_path):

    audio, sample_rate = sf.read(audio_path)

    if len(audio.shape) == 1:
        waveform = torch.tensor(
            audio,
            dtype=torch.float32
        ).unsqueeze(0)
    else:
        waveform = torch.tensor(
            audio.T,
            dtype=torch.float32
    )

    diarization = pipeline({
        "waveform": waveform,
        "sample_rate": sample_rate
    })

    speakers = []

    annotation = diarization.speaker_diarization

    for segment, _, speaker in annotation.itertracks(yield_label=True):
        speakers.append({
            "speaker": speaker,
            "start": float(segment.start),
            "end": float(segment.end)
        })

    return speakers