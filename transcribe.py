from dotenv import load_dotenv
from time import sleep, time
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, Microphone
from typing import Callable, Optional

# Add callback handler
sentence_callback: Optional[Callable[[str], None]] = None

def set_sentence_callback(callback: Callable[[str], None]):
    """Set callback function to handle transcribed sentences"""
    global sentence_callback
    sentence_callback = callback

load_dotenv()

# Global variables
buffer_text = ""
last_flush_time = time()

def start_deepgram_stream():
    try:
        # Initialize Deepgram client
        deepgram = DeepgramClient()

        # Create connection
        dg_connection = deepgram.listen.websocket.v("1")

        def on_message(_, result):
            global buffer_text, last_flush_time, sentence_callback
            try:
                sentence = result.channel.alternatives[0].transcript
                if len(sentence) == 0:
                    return

                now = time()

                if not result.is_final:
                    # Buffer interim results
                    buffer_text = sentence

                    # If 200ms passed, flush
                    if now - last_flush_time > 0.2:  # 200 ms
                        if buffer_text.strip():
                            # print(f"[Chunk] {buffer_text.strip()}")
                            if sentence_callback:
                                sentence_callback(buffer_text.strip())
                        last_flush_time = now
                        buffer_text = ""
                else:
                    # Handle final sentences
                    print(f"[Final] {sentence.strip()}")
                    if sentence_callback:
                        sentence_callback(sentence.strip())
                    buffer_text = ""
                    last_flush_time = now

            except Exception as e:
                print(f"Error in on_message: {e}")

        def on_speech_started(_, **kwargs):
            print("Speech Started")

        def on_utterance_end(_, **kwargs):
            print("Utterance End")

        def on_error(_, error):
            print(f"Error: {error}")

        # Attach listeners
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        # Configure live options
        options = LiveOptions(
            model="nova-3",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            utterance_end_ms=1000,
            vad_events=False,         # Use Deepgram VAD
            # endpointing is NOT set when vad_events=True
        )

        addons = {"no_delay": "true"}

        # Start connection
        if not dg_connection.start(options, addons=addons):
            print("Failed to connect to Deepgram")
            return

        # Open microphone
        microphone = Microphone(dg_connection.send)
        microphone.start()

        print("Recording... Press Enter to stop.")
        input()

        # Finish
        microphone.finish()
        dg_connection.finish()

        print("Streaming finished.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    start_deepgram_stream()
