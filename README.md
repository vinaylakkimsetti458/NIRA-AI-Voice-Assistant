# 🎙️ NIRA – Voice-Based AI Assistant for Analytas

NIRA is a professional **AI voice assistant** built using **LangChain, Whisper, Groq LLM, and FAISS**, designed to provide accurate and conversational responses based solely on the **Agenta documentation**.  
It records speech, transcribes it to text, retrieves the most relevant information, and speaks back the answer in a clear, natural human-like voice.

---

## 🚀 Features

- 🎧 **Voice Interaction:** Speak to NIRA using your microphone.  
- 🧠 **Intelligent Retrieval:** Uses LangChain + FAISS to fetch accurate information from Agenta docs.  
- 💬 **Contextual Responses:** Powered by Groq LLM for smart, structured answers.  
- 🔊 **Natural TTS Output:** Converts responses to speech using Google TTS and Pydub.  
- 📝 **Automatic Transcription:** Whisper converts your voice input into text with high accuracy.  
- ⚙️ **Keyboard Control:** Press keys to start/stop recording or end the session.  

---

## 🧩 Tech Stack

- **LLM Backend:** Groq LLM (`gpt-oss-120b`)  
- **Vector Store:** FAISS with HuggingFace Embeddings  
- **Audio Processing:** PyAudio, SoundDevice, Pydub  
- **Speech Models:** Whisper (OpenAI), gTTS  
- **Frameworks:** LangChain, dotenv  
- **Documentation Source:** [Agenta Docs](https://docs.agenta.ai/)

---

## 🛠️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/nira-voice-assistant.git
cd nira-voice-assistant
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # on Windows
source venv/bin/activate  # on macOS/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Create a `.env` File
Create a `.env` file in the root folder and add:
```
GROQ_API_KEY=your_api_key_here
```

---

## 🎤 Usage

1. Run the script:
   ```bash
   python nira.py
   ```

2. When prompted:
   - Press **`y`** → Start a voice session  
   - Press **`s`** → Start recording your question  
   - Press **`e`** → Stop recording  
   - Press **`q`** → Quit the program  

3. NIRA will:
   - Transcribe your speech using **Whisper**
   - Retrieve the most relevant content from **Agenta Docs**
   - Generate a professional response using **Groq LLM**
   - Speak the response aloud via **Google TTS**

---

## 📂 Project Structure

```
nira/
│
├── nira.py                  # Main script
├── audio/                   # Stores input/output audio
├── .env                     # Environment variables (API key)
├── requirements.txt          # Dependencies
└── README.md                # Documentation
```

---

## ⚡ Future Improvements

- Add streaming responses for real-time conversations  
- Support multi-language voice input/output  
- Deploy NIRA as a web app using Streamlit or FastAPI  

---

## 🙏 Acknowledgements

- [LangChain](https://www.langchain.com/)  
- [Whisper by OpenAI](https://github.com/openai/whisper)  
- [Groq Cloud](https://groq.com/)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- [Agenta Docs](https://docs.agenta.ai/)

---

**💡 Author:** [Vinay Lakkimsetti]  
**📅 Year:** 2025  
**🔗 Repository:** [GitHub Repo Link]
