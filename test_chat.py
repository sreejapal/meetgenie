from chat_with_meeting import ask_meeting_question

transcript = """
Speaker 01: Deployment deadline is June 30.

Speaker 02: Frontend will be completed by June 25.

Speaker 01: Rahul is responsible for deployment.
"""

answer = ask_meeting_question(
    transcript,
    "Who is responsible for deployment?"
)

print(answer)