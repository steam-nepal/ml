from manim_voice import SpeechManager, SpeechServer

speech_manager = SpeechManager(
    engine="google",
    language="en",
    voice="en-US-Wavenet-D",
    speed=1.0,
    pitch=1.0,
    service_settings={
        "credentials_json": "/path/to/google/credentials.json",
    },
)