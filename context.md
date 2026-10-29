# Development Context and Technical Constraints

## Golden Rules for the AI (System Prompts)
1. **Clean, Modular Code:** Do not mix audio-processing logic with the Streamlit interface. Everything must be implemented as pure functions in the backend.
2. **Audio Handling:** Audio captured by the microphone must always be normalized to 16 kHz and a mono channel before being sent to the Hugging Face model.
3. **No Complex External Dependencies:** Use ChromaDB in local mode (in-memory or persisted to the local `./chroma_db` directory). Do not use Docker or cloud services for the database.
4. **Thorough Validation:** Before returning audio-alignment code, ensure that it handles exceptions (for example, empty audio, extreme background noise, or text that does not match at all).

## Exact Data Flow
Audio (.wav) + Expected Text -> Service 1 (Alignment / GOP Score) -> List of Failed Words -> Service 2 (Query ChromaDB) -> Extracted Rule -> LLM Orchestrator -> UI.
