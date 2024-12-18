import whisper
import pyttsx3
import pyaudio
import wave
import time
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify,send_from_directory
import os
import uuid  # To generate unique IDs for each chat room

genai.configure(api_key="AIzaSyB09q12pwTfPUS2suJ3LhRN6GgvYs1lwEE")

# Load Whisper model
whisper_model = whisper.load_model("tiny")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
options = whisper.DecodingOptions(fp16=False)
app = Flask(__name__)

# Initialize text-to-speech engine (pyttsx3)
engine = pyttsx3.init()

# Parameters for recording
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1               # Number of audio channels (1 for mono)
RATE = 44100               # Sampling rate (samples per second)
CHUNK = 1024               # Buffer size (number of frames per buffer)
RECORD_SECONDS = 5         # Duration of recording in seconds

def record_audio():
    """Records audio from the microphone for a specified duration."""
    audio = pyaudio.PyAudio()
    
    # Start recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording complete.")

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    with wave.open("user_input.wav", 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def get_response(input_text):
    try:
        # Start a new chat session with the AI model
        chat = gemini_model.start_chat(history=[])
        response = chat.send_message(input_text + "in short")
        print("AI Response:", response.text)
        return response.text
    except Exception as e:
        print("Error generating AI response:", str(e))
        return "Error generating AI response."

def speech_to_text():
    """Converts speech in an audio file to text using Whisper."""
    audio_file = "user_input.wav"  # Use the recorded audio file
    result = whisper_model.transcribe(audio_file)
    return result["text"]

def text_to_speech(text, filename):
    """Converts text to speech using pyttsx3 and saves it to a file."""
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 150)
    engine.save_to_file(text, filename)
    engine.runAndWait()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Save the audio file temporarily
    temp_file_path = 'user_input.wav'
    audio_file.save(temp_file_path)

    # Transcribe the audio to text using Whisper
    result = whisper_model.transcribe(temp_file_path)
    
    return jsonify({'response': result['text']})

@app.route('/get_response', methods=['POST'])
def speech_to_speech_system():
    # Generate a unique folder for each chat room
    chat_room_id = str(uuid.uuid4())  # Generate a unique identifier for the chat room
    chat_room_folder = f"./chat_rooms/{chat_room_id}"  # Define the folder path for the chat room

    os.makedirs(chat_room_folder, exist_ok=True)  # Create the folder if it doesn't exist

    user_input = request.json.get('input_text')  # Get text input from the request body
    
    if not user_input:
        return jsonify({'error': 'No input text provided'}), 400
    
    print(f"User input: {user_input}")
    audio_path = f"{chat_room_folder}/output.mp3"
    
    response = get_response(user_input)  # Get AI response based on user input
    text_to_speech(response, f"{chat_room_folder}/output.mp3")  # Save the AI response to the chat room folder
    
    return jsonify({'response': response,
                    'audio_path': f"{chat_room_folder}/output.mp3"})  # Return the response as JSON

@app.route('/chat_rooms/<room_id>/<filename>')
def serve_audio(room_id, filename):
    directory = f'chat_rooms/{room_id}'
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    app.run(debug=True)
