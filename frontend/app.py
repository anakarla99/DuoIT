"""Streamlit application for pronunciation and dramatic coaching."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from backend.knowledge.semantic_search import (
    PhoneticSearchResult,
    search_phonetic_rules,
)
from backend.scorer.pronunciation_scoring import (
    PronunciationScoreResult,
    score_pronunciation,
)
from backend.scorer.transcription import (
    TranscriptionResult,
    create_whisper_pipeline,
    transcribe_audio,
)
from core.drama_coach import (
    DramaCoachRequest,
    DramaCoachResponse,
    create_local_coach_pipeline,
    generate_coaching,
)

load_dotenv()


@st.cache_resource(show_spinner="Loading the speech-recognition model...")
def get_speech_recognizer(model_name: str):
    """Cache the Whisper pipeline for the Streamlit process."""
    return create_whisper_pipeline(model_name)


@st.cache_resource(show_spinner="Loading the dramatic-coach model...")
def get_coach_generator(model_name: str):
    """Cache the local text-generation pipeline for the Streamlit process."""
    return create_local_coach_pipeline(model_name)


def analyze_recording(
    audio_file,
    expected_text: str,
    acting_direction: str,
) -> tuple[
    TranscriptionResult,
    PronunciationScoreResult,
    tuple[PhoneticSearchResult, ...],
    DramaCoachResponse,
]:
    """Run the complete local analysis pipeline."""
    if not expected_text.strip():
        raise ValueError("Expected text cannot be empty.")
    if not acting_direction.strip():
        raise ValueError("Acting direction cannot be empty.")

    asr_model = os.getenv("ASR_MODEL_NAME", "openai/whisper-small")
    coach_model = os.getenv("COACH_MODEL_NAME", "distilgpt2")
    transcription = transcribe_audio(
        audio_file,
        model_name=asr_model,
        recognizer=get_speech_recognizer(asr_model),
    )
    score_result = score_pronunciation(expected_text, transcription)

    rules: tuple[PhoneticSearchResult, ...] = ()
    if score_result.failed_words:
        query = " ".join(score_result.failed_words)
        rules = tuple(
            search_phonetic_rules(
                query,
                language="en",
                database_path=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
                collection_name=os.getenv(
                    "CHROMA_COLLECTION_NAME",
                    "phonetic_rules",
                ),
            )
        )

    response = generate_coaching(
        DramaCoachRequest(
            expected_text=expected_text,
            score_result=score_result,
            rules=rules,
            acting_direction=acting_direction,
        ),
        model_name=coach_model,
        generator=get_coach_generator(coach_model),
    )
    return transcription, score_result, rules, response


def render_score_table(score_result: PronunciationScoreResult) -> None:
    """Render word-level score details."""
    st.subheader("Word scores")
    st.dataframe(
        [
            {
                "Expected": word.expected_word,
                "Recognized": word.recognized_word or "(missing)",
                "Score": round(word.score, 3),
                "Passed": word.passed,
                "Start (s)": word.start_seconds,
                "End (s)": word.end_seconds,
            }
            for word in score_result.words
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="CAPT & Drama Coach AI",
        page_icon="🎭",
        layout="wide",
    )
    st.title("CAPT & Drama Coach AI")
    st.caption("Practice pronunciation, then turn corrections into performance.")

    expected_text = st.text_area(
        "Expected text",
        placeholder="Enter the text you want to practice...",
        height=120,
    )
    acting_direction = st.text_input(
        "Acting direction",
        value="Speak clearly and confidently.",
    )
    st.subheader("Recording")
    recorded_audio = st.audio_input(
        "Record your reading",
        help="Allow microphone access, record your reading, then stop recording.",
    )
    uploaded_audio = st.file_uploader(
        "Or upload a recording",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
    )
    audio_file = recorded_audio or uploaded_audio

    if audio_file is not None:
        st.audio(audio_file)

    if not st.button("Analyze recording", type="primary"):
        return
    if audio_file is None:
        st.error("Record your reading or upload an audio recording first.")
        return

    try:
        with st.spinner("Transcribing, scoring, and generating coaching..."):
            audio_file.seek(0)
            transcription, scores, rules, coaching = analyze_recording(
                audio_file,
                expected_text,
                acting_direction,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        st.error(f"Analysis failed: {error}")
        return

    st.subheader("Transcription")
    st.write(transcription.text)
    passed = sum(word.passed for word in scores.words)
    total = len(scores.words)
    st.metric("Words passing threshold", f"{passed}/{total}")
    render_score_table(scores)

    if scores.failed_words:
        st.subheader("Phonetic guidance")
        for rule in rules:
            st.markdown(
                f"**{rule.metadata.get('title', rule.rule_id)}**  \n"
                f"{rule.metadata.get('coaching_tip', rule.document)}"
            )
    else:
        st.success("All expected words passed the pronunciation threshold.")

    st.subheader("Drama coach")
    st.write(coaching.feedback)


if __name__ == "__main__":
    main()
