import streamlit as st
import sounddevice as sd
import numpy as np
import io
import time
import threading
from pydub import AudioSegment
from pydub.playback import play
from elevenlabs import ElevenLabs
import webrtcvad

# App UI setup
st.title("🎤 Real-Time Celebrity Voice Changer with Pause Detection")

# Session state to control UI
if "recording_started" not in st.session_state:
    st.session_state.recording_started = False

# Stop flag shared between UI and thread
stop_flag = threading.Event()
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the environment variables

api_key = os.getenv("ELEVENLABS_API_KEY")  # This should now get the value from .env
print(api_key)  # Should print the API key if set correctly



user_api_key = st.text_input("Enter your ElevenLabs API Key (optional):", type="password")
api_key = user_api_key if user_api_key else os.getenv("ELEVENLABS_API_KEY")
print(os.getenv("ELEVENLABS_API_KEY"))
client = ElevenLabs(api_key=api_key)

# Voice options
available_voices = {
    "Laura (Default)": "FGY2WhTYpPnrIDTdsKH5",
    "Roger": "CwhRBWXzGAHq8TQ4Fs17",
    "River": "SAz9YHcvj6GT2YYXdXww",
    "Matilda": "VoiceXrExE9yKIg1WjnnlVkGXID_3"
}
selected_voice = st.selectbox("Choose a voice:", list(available_voices.keys()))
voice_id = available_voices[selected_voice]

# VAD setup
vad = webrtcvad.Vad(3)

def is_speech(audio_chunk, sample_rate):
    try:
        return vad.is_speech(audio_chunk.tobytes(), sample_rate)
    except:
        return False

def process_audio(buffer_audio, fs):
    if not buffer_audio:
        return
    audio_data = np.concatenate(buffer_audio, axis=0)
    audio_bytes = io.BytesIO()

    segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=fs,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )
    segment = segment.normalize()
    segment.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    try:
        response_gen = client.speech_to_speech.convert_as_stream(
            voice_id=voice_id,
            output_format="mp3_44100_128",
            audio=audio_bytes,
            model_id="eleven_english_sts_v2",
            remove_background_noise=True
        )
        response_bytes = b"".join(response_gen)
        if not stop_flag.is_set():
            play(AudioSegment.from_file(io.BytesIO(response_bytes), format="mp3"))
    except Exception as e:
        print(f"Error processing audio: {e}")

def record_loop():
    fs = 16000
    buffer_audio = []
    silence_duration = 0.0

    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
            while not stop_flag.is_set():
                audio_chunk = stream.read(int(0.03 * fs))[0]
                if is_speech(audio_chunk, fs):
                    buffer_audio.append(audio_chunk)
                    silence_duration = 0.0
                else:
                    silence_duration += 0.03

                if silence_duration >= 2 and buffer_audio:
                    process_audio(buffer_audio, fs)
                    buffer_audio = []
    except Exception as e:
        print(f"Recording error: {e}")
    finally:
        st.session_state.recording_started = False

# UI buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Recording"):
        if not st.session_state.recording_started:
            stop_flag.clear()
            threading.Thread(target=record_loop, daemon=True).start()
            st.session_state.recording_started = True
            st.success("Recording started!")

with col2:
    if st.button("⏹️ Stop Recording"):
        stop_flag.set()
        st.session_state.recording_started = False
        st.warning("Recording stopped. Refresh the page to reset.")




