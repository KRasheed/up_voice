# Real-Time Voice Changer

A Python application that provides real-time voice transformation using speech-to-text (STT) and text-to-speech (TTS) technologies. The application captures your voice, converts it to text using Deepgram's Nova-3 model, and then converts it back to speech using Cartesia's Sonic-2 TTS with a different voice.

## Features

- **Real-time voice processing** with low latency
- **Smart text processing** to avoid duplicate audio and handle speech corrections
- **Configurable voice selection** through Cartesia's voice library
- **Detailed logging** with adjustable verbosity levels
- **Thread-safe audio processing** for smooth playback
- **Intelligent speech session management** to handle pauses and new utterances

## How It Works

1. **Speech Recognition**: Captures audio from your microphone using Deepgram's live transcription
2. **Text Processing**: Intelligently processes both interim and final transcription results
3. **Voice Synthesis**: Converts processed text to speech using Cartesia's TTS API
4. **Audio Playback**: Streams the synthesized audio in real-time

## Prerequisites

- Python 3.7+
- A Deepgram API account and API key
- A Cartesia API account and API key


## Configuration

### Key Configuration Options

```python
# Voice and API Configuration
VOICE_ID = "a0e99841-438c-4a64-b679-ae501e7d6091"  # Cartesia voice ID
WEBSOCKET_URL = 'wss://api.cartesia.ai/tts/websocket?cartesia_version=2025-04-16&api_key=YOUR_API_KEY'

# Processing Parameters
MIN_SEGMENT_LENGTH = 5        # Minimum text length to process
SPEECH_TIMEOUT = 2.5          # Seconds before starting new session
SIMILARITY_THRESHOLD = 0.90   # Duplicate detection threshold
LOG_LEVEL = "NORMAL"          # Logging verbosity: MINIMAL, NORMAL, VERBOSE
```

### Audio Format Settings

The application uses optimized audio settings for real-time processing:
- **Sample Rate**: 8kHz (for TTS output)
- **Encoding**: 16-bit PCM
- **Channels**: Mono
