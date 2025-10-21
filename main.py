import os
import re
import warnings
import pyaudio
import keyboard
import wave
import whisper
import numpy as np
import sounddevice as sd
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Suppress FP16 warning from Whisper
warnings.filterwarnings("ignore",message = "FP16 is not supported on CPU,using FP32 instead")

# Load environment variables from .env file
load_dotenv()

# checking whether the API Key lis loaded or not
API_KEY = os.getenv('GROQ_API_KEY')
if not API_KEY:
    raise ValueError('GROQ API KEY not found in .env file')

# System prompt for Nira
SYSTEM_PROMPT = """
Your name is Nira.
You are the AI assistant for Analytas, an AI advisory firm specializing in the safe and intelligent deployment of autonomous AI agents for organizations.
You use information only from the Agenta documentation (https://docs.agenta.ai/).
Start by asking what brought the user to Analytas today—whether they’re curious about AI agents, checking readiness, or want to schedule a discovery call with the team (requesting their full name, email, phone, date, time, and timezone).
Respond only to questions related to AI agents and Analytas' services as described in the Agenta documentation. Keep answers clear, concise, and focused, avoiding speculation or topics outside the documentation. Your tone must be professional, trustworthy, and conversational, reflecting thoughtful expertise..
Keep your tone professional, trustworthy, and conversational.
If a topic is outside Agenta docs, say exactly:
“Great question. I don't have a confident answer on that just yet — but we're always learning. If it’s important, I’d recommend scheduling a short discovery call with our team.”
If the user seems ready or unsure, gently offer to schedule a discovery call to explore how Analytas can support their goals. Your goal is to build trust and provide accurate information about AI agents.
At the end of each response, conclude naturally with a polite closing such as “Thank you.”
"""

# File to store FAISS vectorstoredb
DOCS_FILE = "agenta_docs_faiss"

# Function to create FAISS vectorstoredb
def load_or_create_index():
    # Checking whether the file exists or not
    if os.path.exists(DOCS_FILE):
        print('Loading cached FAISS index')
        # Creating FAISS vectorstore db
        faiss_db = FAISS.load_local(
            DOCS_FILE,
            HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2'),  # Initializing HuggingFace Embeddings
            allow_dangerous_deserialization = True
        )
        return faiss_db
    
    print('Scraping all Agenta documentation pages...')
    # Scraping the data from the website
    loader = WebBaseLoader('https://docs.agenta.ai/')
    documents = loader.load()

    # Split the data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

    # Create FAISS vectorstore db
    faiss_db = FAISS.from_documents(docs,embeddings)

    # Save the file
    faiss_db.save_local(DOCS_FILE)
    print(f"✅ Scraped and embedded {len(docs)} document chunks.")
    return faiss_db

# Initializing Retrieval Chain
def build_qa_chain(faiss_db):
    # Initializing the Groq LLM
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=API_KEY)

    # Creating retriver
    retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Creating retrieval chain
    chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        chain_type = 'stuff',
        verbose = False
    )
    return chain

# getting the clean answer from generated answer
def clean_response(text):
    """
    Clean AI response for TTS:
    - Remove unwanted symbols like *, -, /, _, etc.
    - Keep punctuation for natural speech.
    - Normalize spaces.
    """
    # Remove markdown-like syntax and unwanted characters
    text = re.sub(r"[*_`#<>\\/\-]+", "", text)   # Removes *, /, -, _, \, etc.

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Correct common transcription errors
def correct_transcription(text):
    corrections = {
        "Nera": "Nira",
        "Neera": "Nira",
        "A.A agents": "AI agents",
        "A A agents": "AI agents",
        "Motor AI Agents": "What are AI agents",
        "soped": "speed",
        "voicfe": "voice",
        "humkan generated voicfe": "human-generated voice"
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

# Audio recording function with keyboard control
AUDIO_PATH = 'audio/input.wav'

def record_audio_with_keyboard(path = AUDIO_PATH):
    print("Press 's' to start recording, 'e' to stop...")

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    # Initializing the PyAudio Interface object(Connects program with computer's microphone and speakers via PortAudio)
    p = pyaudio.PyAudio()
    stream = None
    frames = []

    try:
        # Wait for 's' key to start
        keyboard.wait('s')
        print("Recording started...")

        # Opening the stream to record
        stream = p.open(
            format = FORMAT,
            channels = CHANNELS,
            rate = RATE,
            input = True,
            frames_per_buffer = CHUNK
        )

        # Record until 'e' key is pressed
        while not keyboard.is_pressed('e'):
            # Reading the audio samples in the stream
            data = stream.read(
                CHUNK,
                exception_on_overflow = False)
            frames.append(data)
        print('Recording stopped.')

    # Stop and close stream
    finally:
        if stream:
            stream.stop_stream
            stream.close()
        p.terminate()

    # Save audio
    os.makedirs('audio',exist_ok = True)
    with wave.open(path,'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print('Saved : ',path)

# Transcription function
def transcribe_whisper(path = AUDIO_PATH):
    print('Transcribing...')
    try:
        model = whisper.load_model("small",device = "cpu")
    except:
        print('Small model unavailable,falling back to base model.')
        model = whisper.load_model("base",device = "cpu")
    
    # Converting speech to text using whisper model
    result = model.transcribe(path,language = "en")
    print("You said :",result["text"])
    return result["text"]

# TTS function with faster speed and human-like enhancements
def text_to_speech(text,output_path = 'audio/output.wav'):
    print("Generating audio response...")
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Initializing empty AudioSegment object to append all TTS generated sentences
        combined_audio = AudioSegment.empty()
        for sentence in sentences:
            if sentence:
                # Converting text to speech using gTTS
                tts = gTTS(
                    text = sentence,
                    lang = 'en',
                    tld = 'co.uk',
                    slow = False
                    )
                
                #Initializing the default path for temp_mp3
                temp_mp3 = "audio/temp_output.mp3"

                # Creating a audio named folder
                os.makedirs('audio',exist_ok = True)

                # Saving the temp_mp3 file
                tts.save(temp_mp3)

                # Converting the mp3 file into an AudioSegment object for further manipulation
                audio = AudioSegment.from_mp3(temp_mp3)

                # Increasing the speed
                audio = audio.speedup(playback_speed = 1.2)

                # Enhancing the voice
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * 1.059 ** 2)
                })

                # Appends the processed audio of this sentence to combined_audio
                combined_audio += audio + AudioSegment.silent(duration = 200)

                # Deleting the temporarily saved file
                os.remove(temp_mp3)

        # Saves the full concatenated audio as a .wav file
        combined_audio.export(output_path,format = 'wav')
        print(f"Audio response saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating TTS: {e}")
        return None
    
# Play audio function
def play_audio(file_path):
    print('Playing Audio Response...')
    try:
        # Open the WAV file
        with wave.open(file_path,'rb') as wf:
            sr = wf.getframerate()     # Get sample rate
            data = wf.readframes(wf.getnframes())     # Read all audio frames
            audio = np.frombuffer(data,dtype = np.int16)     # Convert bytes to NumPy array

        # Play the audio
        sd.play(audio,sr)
        sd.wait()
        print('Audio playback complete.')
    except Exception as e:
        print(f"Error playing audio: {e}")

# Main Function
if __name__ == "__main__":
    print("NIRA Voice Assistant (Production-Ready LangChain Edition)")
    faiss_db = load_or_create_index()
    qa_chain = build_qa_chain(faiss_db)
    print("✅ Nira is ready! Press 'y' to talk or 'q' to quit.")

    while True:
        event = keyboard.read_event(suppress = True)
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'q':
                print('Goodbye!')
                break
            elif event.name == 'y':
                # Record audio
                record_audio_with_keyboard()

                # Transcribe
                query = transcribe_whisper()
                query = correct_transcription(query)

                # Generate response with LangChain RAG
                full_prompt = f"{SYSTEM_PROMPT}\nUser: {query}"
                response = qa_chain.run(full_prompt)

                # Getting the cleaned answer from the generated response
                cleaned_response = clean_response(response)
                print("NIRA : ",cleaned_response)

                # Convert to speech
                audio_file = text_to_speech(cleaned_response)

                if audio_file:
                # Playing the audio
                    play_audio(audio_file)
                print("\nPress 'y' to ask another question or 'q' to quit.")
