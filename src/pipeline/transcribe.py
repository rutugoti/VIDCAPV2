import subprocess
import os
from openai import OpenAI
from src.config import GROQ_API_KEY, SPEECH_MODEL

def extract_and_transcribe_audio(video_path, output_dir):
    """
    Extracts audio to mp3 and transcribes using Groq Whisper.
    Returns transcript string or None.
    """
    audio_path = os.path.join(output_dir, "audio.mp3")
    
    # Extract audio track
    cmd = f"ffmpeg -y -i \"{video_path}\" -vn -acodec libmp3lame -q:a 4 \"{audio_path}\""
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        return None
        
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    
    try:
        with open(audio_path, "rb") as audio_file:
            transcript_obj = client.audio.transcriptions.create(
                model=SPEECH_MODEL,
                file=audio_file,
                response_format="text"
            )
        transcript = transcript_obj.strip()
        hallucinations = ["thank you", "subscrib", "watching", "captioned by", "amara"]
        if any(phrase in transcript.lower() for phrase in hallucinations) or len(transcript.split()) < 3:
            return None
        return transcript
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        return None
