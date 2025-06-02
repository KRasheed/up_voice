import asyncio
import json
import uuid
import websockets
import base64
import pyaudio
import queue
import threading
import time
import difflib
import re
from typing import Optional, Dict, List, Set
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, Microphone

# Load environment variables
load_dotenv()

# -------------------- Cartesia TTS Configuration --------------------
VOICE_ID = "a0e99841-438c-4a64-b679-ae501e7d6091"  # Replace with your preferred voice
WEBSOCKET_URL = 'wss://api.cartesia.ai/tts/websocket?cartesia_version=2025-04-16&api_key=YOUR_API_KEY'  # Replace with your API key
OUTPUT_FORMAT = {
    "container": "raw",
    "encoding": "pcm_s16le",
    "sample_rate": 8000,
}

# -------------------- Global State --------------------
tts_websocket = None
context_id = str(uuid.uuid4())
audio_queue = queue.Queue()
sentence_queue = queue.Queue()
playback_finished = threading.Event()
is_running = True
main_event_loop = None
tts_enabled = True
tts_in_progress = False  # Flag to track if TTS is currently processing

# Speech tracking
last_processed_text = ""      # Last text that we've processed
processed_segments = set()    # Set of text segments we've already processed
last_sent_time = 0            # When we last sent text to TTS
last_speech_time = 0          # When we last received any speech data
current_session_id = None     # Current speech session ID

# Configuration
MIN_SEGMENT_LENGTH = 5        # Minimum length of a segment to send to TTS
SPEECH_TIMEOUT = 2.5          # Seconds of silence before starting a new session
SIMILARITY_THRESHOLD = 0.90   # Threshold for considering texts similar (0-1)
LOG_LEVEL = "NORMAL"          # Logging level: "MINIMAL", "NORMAL", "VERBOSE"

# -------------------- Logging Functions --------------------
def log(message, level="NORMAL"):
    """Log messages with appropriate level filtering"""
    if level == "MINIMAL":
        print(message)
    elif level == "NORMAL" and LOG_LEVEL in ["NORMAL", "VERBOSE"]:
        print(message)
    elif level == "VERBOSE" and LOG_LEVEL == "VERBOSE":
        print(f"[DEBUG] {message}")

# -------------------- Helper Functions --------------------
def text_normalize(text):
    """Normalize text for comparison (lowercase, remove extra spaces and punctuation)"""
    # Convert to lowercase and remove punctuation
    normalized = ''.join(c.lower() for c in text if c.isalnum() or c.isspace())
    # Remove extra spaces
    return ' '.join(normalized.split())

def are_texts_similar(text1, text2):
    """Determine if two texts are similar enough to be considered duplicates"""
    if not text1 or not text2:
        return False
    
    # Normalize texts for comparison
    text1_norm = text_normalize(text1)
    text2_norm = text_normalize(text2)
    
    # For very short texts, check for exact match after normalization
    if len(text1_norm) < 10 or len(text2_norm) < 10:
        return text1_norm == text2_norm
    
    # For longer texts, use sequence matcher
    similarity = difflib.SequenceMatcher(None, text1_norm, text2_norm).ratio()
    log(f"Similarity between '{text1}' and '{text2}': {similarity:.2f}", "VERBOSE")
    
    return similarity >= SIMILARITY_THRESHOLD

def extract_new_content(existing, new_text):
    """
    Extract only the new content from new_text compared to existing.
    Ensures word boundaries are respected.
    """
    if not existing:
        return new_text
    
    # If new text is shorter, there's nothing new
    if len(new_text) <= len(existing):
        return ""
    
    # If the new text doesn't broadly continue from the existing text,
    # it's likely a correction or different phrase, so return the whole thing
    if not new_text.lower().startswith(existing.lower()[:min(len(existing), 5)]):
        return new_text
    
    # Get the raw addition
    addition = new_text[len(existing):]
    
    # If the existing text doesn't end with a space and addition doesn't start with one,
    # we need to find a word boundary
    if existing and not existing.endswith(" ") and addition and not addition.startswith(" "):
        # Find the last word in existing_text
        last_word_match = re.search(r'(\S+)$', existing)
        if last_word_match:
            last_word = last_word_match.group(1)
            
            # Adjust the addition to include the full word
            return last_word + addition
    
    return addition

def is_segment_already_processed(text):
    """Check if a text segment or something very similar is already processed"""
    global processed_segments
    
    for processed in processed_segments:
        if are_texts_similar(text, processed):
            return True
    return False

def should_start_new_session():
    """Determine if we should start a new speech session based on time elapsed"""
    global last_speech_time
    time_since_speech = time.time() - last_speech_time
    return time_since_speech > SPEECH_TIMEOUT

def split_into_chunks(text, max_chunk_size=50):
    """
    Split text into logical chunks at natural boundaries.
    Prefers splitting at punctuation or on word boundaries.
    """
    # If text is short enough, return as is
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    remaining = text
    
    while remaining:
        # If what's left is short enough, add it and we're done
        if len(remaining) <= max_chunk_size:
            chunks.append(remaining)
            break
        
        # Try to find a good split point within max_chunk_size
        # Priority: sentence end, comma, space from right to left
        chunk_to_search = remaining[:max_chunk_size]
        
        # Look for sentence end
        sentence_end = max(chunk_to_search.rfind('.'), 
                         chunk_to_search.rfind('!'), 
                         chunk_to_search.rfind('?'))
        
        if sentence_end > 0:
            # +1 to include the punctuation mark
            split_point = sentence_end + 1
        else:
            # Look for comma
            comma = chunk_to_search.rfind(',')
            if comma > 0:
                split_point = comma + 1
            else:
                # Look for space
                space = chunk_to_search.rfind(' ')
                if space > 0:
                    split_point = space
                else:
                    # No natural break found, just split at max_chunk_size
                    split_point = max_chunk_size
        
        # Add the chunk and continue with what's left
        chunks.append(remaining[:split_point].strip())
        remaining = remaining[split_point:].strip()
    
    return chunks

# -------------------- Audio Player Thread --------------------
def audio_player_thread():
    """Thread function to play audio chunks as they become available"""
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=8000,
        output=True
    )
    
    try:
        while is_running or not audio_queue.empty():
            try:
                # Get chunk from queue, blocks until something is available with a timeout
                chunk_data = audio_queue.get(timeout=0.5)
                
                # None is our signal to stop
                if chunk_data is None:
                    break
                    
                # Play the chunk
                stream.write(chunk_data)
                audio_queue.task_done()
            except queue.Empty:
                continue
            
    except Exception as e:
        log(f"Error in audio playback: {e}", "MINIMAL")
    finally:
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()
        playback_finished.set()
        log("Audio player thread stopped", "NORMAL")

# -------------------- TTS WebSocket Handler --------------------
async def handle_tts_websocket_responses():
    """Handle incoming audio data from websocket and send to player"""
    global tts_websocket, tts_in_progress
    
    if tts_websocket is None:
        log("WebSocket connection not established", "MINIMAL")
        return
    
    try:
        while is_running:
            try:
                response = await asyncio.wait_for(tts_websocket.recv(), timeout=0.5)
                
                if isinstance(response, bytes):
                    # Ignore raw bytes packets
                    continue

                data = json.loads(response)

                if 'data' in data:
                    audio_data = base64.b64decode(data['data'])
                    # Put raw audio data directly into the queue for playback
                    audio_queue.put(audio_data)

                if data.get("done"):
                    tts_in_progress = False
                    log("TTS chunk completed", "VERBOSE")
            except asyncio.TimeoutError:
                continue
                
    except websockets.exceptions.ConnectionClosed:
        log("TTS WebSocket connection closed", "NORMAL")
        tts_in_progress = False
    except Exception as e:
        log(f"Error processing TTS WebSocket responses: {e}", "MINIMAL")
        tts_in_progress = False

# -------------------- TTS Functions --------------------
async def init_tts_websocket():
    """Initialize the TTS WebSocket connection"""
    global tts_websocket, tts_enabled
    
    try:
        # Print the URL (without the API key for security)
        url_parts = WEBSOCKET_URL.split('api_key=')
        safe_url = url_parts[0] + 'api_key=****'
        log(f"Connecting to TTS WebSocket: {safe_url}", "NORMAL")
        
        tts_websocket = await websockets.connect(WEBSOCKET_URL)
        log("TTS WebSocket connection opened ✅", "MINIMAL")
        
        # Start the response handler
        asyncio.create_task(handle_tts_websocket_responses())
        return True
    except websockets.exceptions.WebSocketException as e:
        if "HTTP 402" in str(e):
            log("⚠️ Payment Required Error: Your Cartesia account has payment issues.", "MINIMAL")
            log("  - Check if your API key is correct", "MINIMAL")
            log("  - Verify your account has sufficient credits", "MINIMAL")
            log("  - Update billing information if needed", "MINIMAL")
            log("\nContinuing with speech recognition only (no voice changing)...", "MINIMAL")
            tts_enabled = False
        else:
            log(f"Error establishing TTS WebSocket connection: {e}", "MINIMAL")
            tts_enabled = False
        return False
    except Exception as e:
        log(f"Error establishing TTS WebSocket connection: {e}", "MINIMAL")
        tts_enabled = False
        return False

async def send_text_to_tts(text, continue_utterance=True):
    """Send text to TTS service with improved handling"""
    global tts_websocket, tts_enabled, tts_in_progress, context_id, last_sent_time
    
    # Update our tracking
    last_sent_time = time.time()
    
    # Skip if TTS is disabled due to connection issues
    if not tts_enabled:
        log(f"TTS disabled, would have sent: '{text}'", "NORMAL")
        return
    
    if tts_websocket is None:
        success = await init_tts_websocket()
        if not success:
            log("Failed to initialize TTS WebSocket", "MINIMAL")
            return
    
    try:
        # Mark that TTS is in progress
        tts_in_progress = True
        
        # Only create a new context if starting a new utterance
        if not continue_utterance:
            context_id = str(uuid.uuid4())
            log(f"New speech context: {context_id}", "NORMAL")
        
        # Send the message
        message = {
            "context_id": context_id,
            "model_id": "sonic-2",
            "voice": {
                "mode": "id",
                "id": VOICE_ID
            },
            "language": "en",
            "output_format": OUTPUT_FORMAT,
            "transcript": text,
            "continue": continue_utterance
        }
        
        await tts_websocket.send(json.dumps(message))
        log(f"Sent to TTS: '{text}' (continue={continue_utterance})", "NORMAL")
        
    except Exception as e:
        log(f"Error sending text to TTS: {e}", "MINIMAL")
        tts_in_progress = False
        # Try to reconnect
        await init_tts_websocket()

# -------------------- Sentence Processing --------------------
async def process_sentence_queue():
    """Process sentences from the queue in the main event loop"""
    global last_processed_text, processed_segments, last_speech_time, current_session_id
    
    while is_running:
        try:
            # Check if we need to start a new session due to silence
            if should_start_new_session() and current_session_id is not None:
                log("Starting new speech session due to silence", "NORMAL")
                current_session_id = None
                last_processed_text = ""
                # Keep processed_segments to prevent repeats across utterances
            
            # Non-blocking check of queue
            if not sentence_queue.empty():
                sentence_data = sentence_queue.get_nowait()
                is_final = sentence_data["is_final"]
                text = sentence_data["text"].strip()
                
                # Update last speech time
                last_speech_time = time.time()
                
                # Skip empty texts
                if not text or len(text) < MIN_SEGMENT_LENGTH:
                    sentence_queue.task_done()
                    continue
                
                # Create a new session if needed
                if current_session_id is None:
                    current_session_id = str(uuid.uuid4())
                    log(f"Starting new speech session: {current_session_id}", "NORMAL")
                
                # PROCESS FINAL RESULTS
                if is_final:
                    # If we've already processed this text or something very similar, skip it
                    if is_segment_already_processed(text):
                        log(f"Skipping duplicate final text: '{text}'", "NORMAL")
                        sentence_queue.task_done()
                        continue
                    
                    # For long sentences, process in meaningful chunks
                    if len(text) > 80:  # A threshold for "long" sentences
                        chunks = split_into_chunks(text)
                        log(f"Split long sentence into {len(chunks)} chunks", "VERBOSE")
                        
                        first_chunk = True
                        for chunk in chunks:
                            # Skip chunks we've already processed
                            if is_segment_already_processed(chunk):
                                log(f"Skipping already processed chunk: '{chunk}'", "VERBOSE")
                                continue
                                
                            # Wait briefly for any in-progress TTS to complete
                            if tts_in_progress:
                                wait_count = 0
                                while tts_in_progress and wait_count < 30:
                                    await asyncio.sleep(0.01)
                                    wait_count += 1
                            
                            # Send this chunk, continuing from previous if not first chunk
                            await send_text_to_tts(chunk, continue_utterance=not first_chunk)
                            log(f"Sent chunk {'' if first_chunk else 'continuation'}: '{chunk}'", "NORMAL")
                            
                            # Add to processed segments
                            processed_segments.add(text_normalize(chunk))
                            first_chunk = False
                    else:
                        # Check if this adds meaningful content to what we've already processed
                        if last_processed_text:
                            new_content = extract_new_content(last_processed_text, text)
                            
                            if new_content and len(new_content) >= MIN_SEGMENT_LENGTH:
                                # Wait briefly for any in-progress TTS to complete
                                if tts_in_progress:
                                    wait_count = 0
                                    while tts_in_progress and wait_count < 30:
                                        await asyncio.sleep(0.01)
                                        wait_count += 1
                                
                                # Send the new content as a continuation
                                await send_text_to_tts(new_content, continue_utterance=True)
                                log(f"Sent new content from final: '{new_content}'", "NORMAL")
                                
                                # Update our state
                                last_processed_text = text
                                processed_segments.add(text_normalize(text))
                            else:
                                log(f"Final result doesn't add significant content: '{text}'", "NORMAL")
                        else:
                            # This is the first content for this session
                            # Wait briefly for any in-progress TTS to complete
                            if tts_in_progress:
                                wait_count = 0
                                while tts_in_progress and wait_count < 30:
                                    await asyncio.sleep(0.01)
                                    wait_count += 1
                            
                            # Start a new utterance with this text
                            await send_text_to_tts(text, continue_utterance=False)
                            log(f"Sent initial final text: '{text}'", "NORMAL")
                            
                            # Update our state
                            last_processed_text = text
                            processed_segments.add(text_normalize(text))
                
                # PROCESS INTERIM RESULTS - for low latency when speaking
                else:
                    # We want to use interim results intelligently:
                    # 1. When starting a new speech session to get immediate feedback
                    # 2. For significant new content not yet processed
                    
                    # Skip if too similar to what we've already processed
                    if is_segment_already_processed(text):
                        log(f"Skipping similar interim result: '{text}'", "VERBOSE")
                        sentence_queue.task_done()
                        continue
                    
                    # If this is a new speech segment (nothing processed yet)
                    # OR if it adds substantial new content to what we last processed
                    if not last_processed_text or len(text) > len(last_processed_text) + MIN_SEGMENT_LENGTH:
                        # If we're continuing and have existing content
                        if last_processed_text:
                            # Extract just what's new
                            new_content = extract_new_content(last_processed_text, text)
                            
                            # Only send if it's substantial enough
                            if new_content and len(new_content) >= MIN_SEGMENT_LENGTH:
                                # Wait briefly for any in-progress TTS to complete
                                if tts_in_progress:
                                    wait_count = 0
                                    while tts_in_progress and wait_count < 30:
                                        await asyncio.sleep(0.01)
                                        wait_count += 1
                                
                                # Send the chunk and continue the utterance
                                await send_text_to_tts(new_content, continue_utterance=True)
                                log(f"Sent additional interim content: '{new_content}'", "NORMAL")
                                
                                # Update our buffer but don't add to processed_segments
                                # (since this is interim and we'll get a final version)
                                last_processed_text = text
                            else:
                                log(f"Interim content not significant enough: '{new_content}'", "VERBOSE")
                        else:
                            # This is the first content for this session
                            # Wait briefly for any in-progress TTS to complete
                            if tts_in_progress:
                                wait_count = 0
                                while tts_in_progress and wait_count < 30:
                                    await asyncio.sleep(0.01)
                                    wait_count += 1
                            
                            # Start a new utterance with this text
                            await send_text_to_tts(text, continue_utterance=False)
                            log(f"Sent initial interim content: '{text}'", "NORMAL")
                            
                            # Update our state but don't add to processed_segments yet
                            last_processed_text = text
                
                sentence_queue.task_done()
            
            # Short sleep to avoid busy waiting
            await asyncio.sleep(0.01)
        except queue.Empty:
            await asyncio.sleep(0.05)  # Wait a bit longer if queue is empty
        except Exception as e:
            log(f"Error processing sentence queue: {e}", "MINIMAL")
            await asyncio.sleep(0.1)  # Wait longer on error

# -------------------- STT Callback --------------------
def stt_callback(text: str, is_final: bool):
    """Callback function for STT results - thread-safe, just adds to queue"""
    if text:
        # Add structured data to the sentence queue
        sentence_queue.put({
            "text": text,
            "is_final": is_final
        })
        log(f"Queued {'final' if is_final else 'interim'} sentence: {text}", "VERBOSE")

# -------------------- STT Functions --------------------
def start_stt():
    """Initialize and start the STT service"""
    try:
        # Initialize Deepgram client
        deepgram = DeepgramClient()

        # Create connection
        dg_connection = deepgram.listen.websocket.v("1")

        def on_message(_, result):
            try:
                sentence = result.channel.alternatives[0].transcript
                if len(sentence.strip()) == 0:
                    return

                if not result.is_final:
                    # For interim results, only forward substantial ones
                    if len(sentence.strip()) >= MIN_SEGMENT_LENGTH:
                        stt_callback(sentence.strip(), False)
                else:
                    # Always handle final sentences
                    stt_callback(sentence.strip(), True)

            except Exception as e:
                log(f"Error in on_message: {e}", "MINIMAL")

        def on_speech_started(_, **kwargs):
            log("Speech Started", "NORMAL")

        def on_utterance_end(_, **kwargs):
            log("Utterance End", "NORMAL")

        def on_error(_, error):
            log(f"Error: {error}", "MINIMAL")

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
            interim_results=True,  # Always get interim results for low latency
            utterance_end_ms=1000,
            vad_events=False,
        )

        addons = {"no_delay": "true"}

        # Start connection
        if not dg_connection.start(options, addons=addons):
            log("Failed to connect to Deepgram", "MINIMAL")
            return None, None

        # Open microphone
        microphone = Microphone(dg_connection.send)
        microphone.start()

        return dg_connection, microphone
    
    except Exception as e:
        log(f"Error starting STT: {e}", "MINIMAL")
        return None, None

# -------------------- Main Application --------------------
async def main():
    global is_running, main_event_loop
    
    # Store the main event loop reference
    main_event_loop = asyncio.get_event_loop()
    
    log("Starting Voice Changer Application...", "MINIMAL")
    
    # Start audio player thread
    player_thread = threading.Thread(target=audio_player_thread)
    player_thread.daemon = True
    player_thread.start()
    
    # Initialize TTS WebSocket
    tts_init_success = await init_tts_websocket()
    if not tts_init_success:
        log("Voice changing functionality disabled. Continuing with speech recognition only.", "MINIMAL")
    
    # Start sentence processing task in main event loop
    sentence_processor = asyncio.create_task(process_sentence_queue())
    
    # Start STT in a separate thread
    dg_connection, microphone = start_stt()
    
    if not dg_connection or not microphone:
        log("Failed to start STT service", "MINIMAL")
        return
    
    log("Voice Changer is active! Speak into your microphone.", "MINIMAL")
    log("Press Enter to stop.", "MINIMAL")
    
    # Wait for user to press Enter
    await asyncio.get_event_loop().run_in_executor(None, input)
    
    # Cleanup
    is_running = False
    
    # Wait for sentence processor to complete
    try:
        await asyncio.wait_for(sentence_processor, timeout=2)
    except asyncio.TimeoutError:
        log("Sentence processor task timeout - forcing shutdown", "NORMAL")
    
    if microphone:
        microphone.finish()
    
    if dg_connection:
        dg_connection.finish()
    
    if tts_websocket:
        await tts_websocket.close()
    
    # Signal player thread to stop and wait for it
    audio_queue.put(None)
    playback_finished.wait(timeout=5)
    
    log("Voice Changer application stopped.", "MINIMAL")

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())