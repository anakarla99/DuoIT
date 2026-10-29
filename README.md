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

## Local Environment Setup

The project targets Python 3.13 and pins its runtime dependencies in
[`requirements.txt`](requirements.txt).

1. Create a virtual environment:

   ```powershell
   py -3.13 -m venv .venv
   ```

2. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the pinned dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Copy [`.env.example`](.env.example) to `.env` and adjust local settings as needed.

The `.env` file and the local `.venv` directory are ignored by Git.
