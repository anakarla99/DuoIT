"""Seed the local ChromaDB collection with foundational English phonetic rules."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from backend.knowledge.init_db import (
    DEFAULT_COLLECTION_NAME,
    initialize_database,
)


@dataclass(frozen=True)
class PhoneticRule:
    """A searchable pronunciation rule and its coaching metadata."""

    rule_id: str
    phoneme: str
    ipa: str
    title: str
    explanation: str
    examples: str
    common_error: str
    coaching_tip: str
    language: str = "en"

    @property
    def document(self) -> str:
        """Return the text ChromaDB uses to create the rule embedding."""
        return (
            f"{self.title}. Phoneme: {self.phoneme}. IPA: {self.ipa}. "
            f"{self.explanation} Examples: {self.examples}. "
            f"Common error: {self.common_error}. "
            f"Coaching tip: {self.coaching_tip}"
        )

    @property
    def metadata(self) -> dict[str, str]:
        """Return scalar metadata suitable for ChromaDB."""
        return {
            "language": self.language,
            "phoneme": self.phoneme,
            "ipa": self.ipa,
            "title": self.title,
            "examples": self.examples,
            "common_error": self.common_error,
            "coaching_tip": self.coaching_tip,
        }


SEED_RULES: tuple[PhoneticRule, ...] = (
    PhoneticRule(
        rule_id="en-th-voiceless",
        phoneme="th",
        ipa="/θ/",
        title="Voiceless TH",
        explanation="Place the tongue lightly between the teeth and release air without vibrating the vocal cords.",
        examples="think, theater, both",
        common_error="Replacing /θ/ with /t/ or /s/.",
        coaching_tip="Hold a tissue in front of the mouth and check for a small puff of air.",
    ),
    PhoneticRule(
        rule_id="en-th-voiced",
        phoneme="th",
        ipa="/ð/",
        title="Voiced TH",
        explanation="Use the same tongue position as voiceless TH, but vibrate the vocal cords.",
        examples="this, mother, breathe",
        common_error="Replacing /ð/ with /d/ or /z/.",
        coaching_tip="Touch the throat while sustaining the sound; it should gently vibrate.",
    ),
    PhoneticRule(
        rule_id="en-r",
        phoneme="r",
        ipa="/ɹ/",
        title="English R",
        explanation="Curl the tongue slightly without touching the roof of the mouth and keep the lips relaxed.",
        examples="red, arrive, very",
        common_error="Trilling the sound or touching the alveolar ridge.",
        coaching_tip="Keep the tongue suspended and prolong the sound before the following vowel.",
    ),
    PhoneticRule(
        rule_id="en-short-i",
        phoneme="i",
        ipa="/ɪ/",
        title="Short I",
        explanation="Use a relaxed, near-close front vowel that is shorter and more centralized than /iː/.",
        examples="ship, sit, live",
        common_error="Pronouncing /ɪ/ as the longer /iː/.",
        coaching_tip="Keep the jaw relaxed and make the vowel brief.",
    ),
    PhoneticRule(
        rule_id="en-long-ee",
        phoneme="ee",
        ipa="/iː/",
        title="Long EE",
        explanation="Raise the front of the tongue close to the hard palate while keeping the lips slightly spread.",
        examples="sheep, seat, leave",
        common_error="Reducing /iː/ to the shorter /ɪ/.",
        coaching_tip="Sustain the vowel and finish with a slight smile shape.",
    ),
    PhoneticRule(
        rule_id="en-short-a",
        phoneme="a",
        ipa="/æ/",
        title="Short A",
        explanation="Open the mouth wide with the tongue low and forward.",
        examples="cat, bad, actor",
        common_error="Replacing /æ/ with /ɛ/ or /ɑː/.",
        coaching_tip="Lower the jaw more than for the vowel in 'bed' and keep the sound forward.",
    ),
    PhoneticRule(
        rule_id="en-schwa",
        phoneme="unstressed vowel",
        ipa="/ə/",
        title="Schwa",
        explanation="Use a short, relaxed mid-central vowel in an unstressed syllable.",
        examples="about, support, banana",
        common_error="Giving every vowel full stress and a strong dictionary vowel quality.",
        coaching_tip="Reduce the vowel and shift emphasis to the stressed syllable.",
    ),
    PhoneticRule(
        rule_id="en-final-consonants",
        phoneme="final consonant",
        ipa="varies",
        title="Final Consonant Release",
        explanation="Keep the final consonant audible so word boundaries remain clear.",
        examples="walked, asked, books",
        common_error="Dropping final /t/, /d/, /k/, or /s/ sounds.",
        coaching_tip="Speak the final consonant lightly but deliberately before pausing or linking.",
    ),
)


def seed_phonetic_rules(
    database_path: str | Path | None = None,
    collection_name: str | None = None,
) -> int:
    """Upsert the built-in rules and return the number of seeded records."""
    collection = initialize_database(database_path, collection_name)
    collection.upsert(
        ids=[rule.rule_id for rule in SEED_RULES],
        documents=[rule.document for rule in SEED_RULES],
        metadatas=[rule.metadata for rule in SEED_RULES],
    )
    return len(SEED_RULES)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for seeding phonetic rules."""
    parser = argparse.ArgumentParser(
        description="Seed the local ChromaDB collection with phonetic rules."
    )
    parser.add_argument("--path", default=None, help="ChromaDB database directory.")
    parser.add_argument(
        "--collection",
        default=None,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    return parser.parse_args()


def main() -> None:
    """Load configuration, seed the collection, and report its status."""
    load_dotenv()
    args = parse_args()
    seeded_count = seed_phonetic_rules(args.path, args.collection)
    collection_name = args.collection or os.getenv(
        "CHROMA_COLLECTION_NAME",
        DEFAULT_COLLECTION_NAME,
    )
    print(f"Seeded {seeded_count} phonetic rules into {collection_name!r}.")


if __name__ == "__main__":
    main()
