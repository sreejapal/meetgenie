from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_NAME = "gemini-2.5-flash"


def ask_meeting_question(
    transcript,
    question
):

    prompt = f"""
You are a meeting assistant.

Use ONLY the meeting transcript below.

Meeting Transcript:
{transcript}

Question:
{question}

If the answer is not found,
say:
'I could not find that information in the meeting transcript.'
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text