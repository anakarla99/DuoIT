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

### Initialize the Knowledge Database

From the project root, run:

```powershell
python -m backend.knowledge.init_db
```

This creates the local `chroma_db` directory and the `phonetic_rules`
collection if they do not already exist. Existing data is preserved.
The collection is intentionally empty until the phonetic seed-data task.

### Preprocess Audio

The audio preprocessing service loads a WAV-compatible audio source and
normalizes it to the model requirements: 16 kHz, mono, finite `float32`
samples.

```python
from backend.scorer.audio_preprocessing import preprocess_audio

audio = preprocess_audio("recording.wav")
print(audio.sample_rate, audio.channels, audio.duration_seconds)
```

It raises `AudioPreprocessingError` for empty, unreadable, non-finite, or
silent audio instead of passing invalid input to the model.

### Seed Phonetic Rules

Seed the local collection with the initial English phonetic rules:

```powershell
python -m backend.knowledge.seed_data
```

The command is idempotent: it uses stable IDs and `upsert`, so running it
again updates the existing rules instead of creating duplicates. The initial
dataset covers TH sounds, English R, vowel contrasts, schwa, and final
consonant release.

### Search Phonetic Rules

Query the seeded rules semantically:

```python
from backend.knowledge.semantic_search import search_phonetic_rules

results = search_phonetic_rules("I replace the TH sound with a T")
for result in results:
    print(result.rule_id, result.metadata["title"], result.distance)
```

Searches run against the local ChromaDB collection and return the closest
rules with their metadata and distance. An optional `language="en"` filter is
available. Empty queries and non-positive limits are rejected explicitly.

### Transcribe Audio with Word Timestamps

The scorer uses Whisper through the Hugging Face Transformers pipeline because
Whisper supports word-level timestamps directly:

```python
from backend.scorer.transcription import transcribe_audio

result = transcribe_audio("recording.wav")
print(result.text)
for word in result.words:
    print(word.word, word.start_seconds, word.end_seconds)
```

The default model is `openai/whisper-small`. The first execution downloads the
model into the local Hugging Face cache. No model is downloaded when the module
is imported. For tests, inject a recognizer into `transcribe_audio`.

### Score Pronunciation

Compare expected text with the timestamped transcription:

```python
from backend.scorer.pronunciation_scoring import score_pronunciation
from backend.scorer.transcription import transcribe_audio

transcription = transcribe_audio("recording.wav")
scores = score_pronunciation("Think clearly", transcription)

for word in scores.words:
    print(word.expected_word, word.score, word.passed)
print(scores.failed_words)
```

The current score is an intelligibility score based on sequential word
matches and Whisper confidence. It is not a true phoneme-level GOP score;
that requires forced alignment and acoustic logits, which will replace this
scoring implementation when that model integration is added.

### Dramatic Coach

Generate local LLM feedback from pronunciation scores and retrieved phonetic
rules:

```python
from core.drama_coach import DramaCoachRequest, generate_coaching

request = DramaCoachRequest(
    expected_text="Think clearly",
    score_result=scores,
    acting_direction="Sound determined but encouraging.",
)
response = generate_coaching(request)
print(response.feedback)
```

The default local model is `distilgpt2`, configurable with
`COACH_MODEL_NAME`. The first real generation downloads that model into the
local Hugging Face cache. Tests can inject a generator and avoid any download.
