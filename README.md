# CAPT & Drama Coach AI

Local application for practicing reading aloud in other languages, receiving precise phonetic corrections (Goodness of Pronunciation - GOP) and dramatic direction advice.

## System Architecture
- `/frontend`: User interface built with Streamlit.
- `/backend/scorer`: Forced-alignment pipeline (Audio-to-Text) using Wav2Vec2/Whisper.
- `/backend/knowledge`: Vector database (ChromaDB) containing phonetic and linguistic rules.
- `/core`: Central orchestrator and LLM prompts.

## Technology Stack
- Python 3.10+
- PyTorch / Hugging Face Transformers
- Librosa (audio processing)
- ChromaDB (local vector database)
- Streamlit (UI)
