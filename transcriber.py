import whisper
import torch
import os
import subprocess
from diarizer import diarize

device = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model("small").to(device)


def convert_video_to_audio(video_path):
    audio_path = "temp_audio.mp3"

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "mp3",
            audio_path
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return audio_path


def transcribe_video(video_path):
    audio_path = None

    try:
        audio_path = convert_video_to_audio(video_path)

        result = model.transcribe(
            audio_path,
            fp16=torch.cuda.is_available()
        )
        diarization = diarize(audio_path)

        segments = []

        for seg in result.get("segments", []):

            segments.append({
                "speaker": find_speaker(
                    seg,
                    diarization
                ),
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            })

        return {
            "text": result.get("text", ""),
            "language": result.get("language", "unknown"),
            "segments": segments,
            "duration": (
                segments[-1]["end"]
                if segments
                else 0
            )
        }

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

def find_speaker(segment, diarization):

    start = segment["start"]
    end = segment["end"]

    midpoint = (start + end) / 2

    for speaker_seg in diarization:

        if (
            speaker_seg["start"]
            <= midpoint
            <= speaker_seg["end"]
        ):
            return speaker_seg["speaker"]

    return "Unknown"