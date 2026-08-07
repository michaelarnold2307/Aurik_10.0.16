"""
Phoneme Classifier

This module classifies IPA phonemes into linguistic categories such as
vowels, consonants, sibilants, etc. Provides detailed classification for
intelligent audio processing decisions.

Key Features:
- IPA → Category mapping
- Sibilant sub-classification (/s/, /z/, /ʃ/, /ʒ/, /tʃ/, /dʒ/)
- Articulation place and manner
- Voicing detection
- Vowel height/backness classification

Author: Aurik Development Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from enum import Enum

from backend.ml.phoneme_aware.logging_config import setup_logger

logger = logging.getLogger(__name__)

logger = setup_logger(__name__)


class PhonemeCategory(Enum):
    """
    Haupt-phoneme categories.

    Categories are organized by articulation manner for consonants
    and height/backness for vowels.
    """

    # Vowels (by height)
    VOWEL_CLOSE = "close_vowel"  # High vowels: /i/, /u/
    VOWEL_MID = "mid_vowel"  # Mid vowels: /e/, /o/, /ə/
    VOWEL_OPEN = "open_vowel"  # Low vowels: /a/, /ɑ/

    # Consonants (by manner)
    PLOSIVE = "plosive"  # Stops: /p/, /t/, /k/, /b/, /d/, /g/
    FRICATIVE = "fricative"  # Fricatives: /f/, /v/, /θ/, /ð/, /h/
    NASAL = "nasal"  # Nasals: /m/, /n/, /ŋ/
    LIQUID = "liquid"  # Liquids: /l/, /r/, /ɹ/
    GLIDE = "glide"  # Semivowels: /w/, /j/
    AFFRICATE = "affricate"  # Affricates: /tʃ/, /dʒ/

    # Sibilants (special category - subset of fricatives/affricates)
    SIBILANT_ALVEOLAR = "sibilant_s"  # /s/, /z/
    SIBILANT_POSTALVEOLAR = "sibilant_sh"  # /ʃ/, /ʒ/
    SIBILANT_AFFRICATE = "sibilant_ch"  # /tʃ/, /dʒ/

    # Special
    SILENCE = "silence"
    BREATH = "breath"
    UNKNOWN = "unknown"


class SibilantType(Enum):
    """
    Detailed sibilant classification.

    Sibilants are consonants with high-frequency hissing/hushing sound.
    Different sibilants have different frequency characteristics and require
    different processing approaches.
    """

    S_VOICELESS = "s"  # /s/ ~ 8000 Hz center
    Z_VOICED = "z"  # /z/ ~ 7500 Hz center
    SH_VOICELESS = "sh"  # /ʃ/ ~ 5000 Hz center
    ZH_VOICED = "zh"  # /ʒ/ ~ 4500 Hz center
    CH_VOICELESS = "ch"  # /tʃ/ ~ 6000 Hz center (affricate)
    JH_VOICED = "jh"  # /dʒ/ ~ 5500 Hz center (affricate)


class ArticulationPlace(Enum):
    """Place of articulation for consonants."""

    BILABIAL = "bilabial"  # Both lips: /p/, /b/, /m/
    LABIODENTAL = "labiodental"  # Lip-teeth: /f/, /v/
    DENTAL = "dental"  # Tongue-teeth: /θ/, /ð/
    ALVEOLAR = "alveolar"  # Tongue-ridge: /t/, /d/, /s/, /z/, /n/, /l/
    POSTALVEOLAR = "postalveolar"  # Behind ridge: /ʃ/, /ʒ/, /tʃ/, /dʒ/
    PALATAL = "palatal"  # Hard palate: /j/
    VELAR = "velar"  # Soft palate: /k/, /g/, /ŋ/
    GLOTTAL = "glottal"  # Glottis: /h/


@dataclass
class PhonemeInfo:
    """
    Detailed phoneme information.

    Attributes:
        phoneme: IPA symbol
        category: Main category (vowel, plosive, etc.)
        is_vowel: True if vowel
        is_consonant: True if consonant
        is_sibilant: True if sibilant
        is_voiced: True if voiced
        sibilant_type: Detailed sibilant type (if applicable)
        place: Articulation place (if consonant)
    """

    phoneme: str
    category: PhonemeCategory
    is_vowel: bool
    is_consonant: bool
    is_sibilant: bool
    is_voiced: bool
    sibilant_type: SibilantType | None = None
    place: ArticulationPlace | None = None


class PhonemeClassifier:
    """
    Classify IPA phonemes into linguistic categories.

    This classifier provides detailed phoneme classification for intelligent
    audio processing. It maps IPA symbols to linguistic categories and
    provides information about articulation, voicing, and special properties.

    Example:
        >>> classifier = PhonemeClassifier()
        >>> info = classifier.classify_detailed('s')
        >>> logger.info(info.category)
        PhonemeCategory.SIBILANT_ALVEOLAR
        >>> logger.info(info.is_sibilant)
        True
        >>> logger.info(info.sibilant_type)
        SibilantType.S_VOICELESS

    Attributes:
        _phoneme_map: Mapping from IPA to PhonemeCategory
        _sibilant_map: Mapping from IPA to SibilantType
        _voicing_map: Mapping from IPA to voicing (True/False)
        _place_map: Mapping from IPA to ArticulationPlace
    """

    # IPA → Category mapping
    _PHONEME_MAPPINGS: dict[str, PhonemeCategory] = {
        # Close vowels
        "i": PhonemeCategory.VOWEL_CLOSE,
        "u": PhonemeCategory.VOWEL_CLOSE,
        "y": PhonemeCategory.VOWEL_CLOSE,
        "ɨ": PhonemeCategory.VOWEL_CLOSE,
        "ʉ": PhonemeCategory.VOWEL_CLOSE,
        "ɯ": PhonemeCategory.VOWEL_CLOSE,
        "ɪ": PhonemeCategory.VOWEL_CLOSE,
        "ʊ": PhonemeCategory.VOWEL_CLOSE,
        # Mid vowels
        "e": PhonemeCategory.VOWEL_MID,
        "o": PhonemeCategory.VOWEL_MID,
        "ə": PhonemeCategory.VOWEL_MID,  # Schwa
        "ɛ": PhonemeCategory.VOWEL_MID,
        "ɔ": PhonemeCategory.VOWEL_MID,
        "ø": PhonemeCategory.VOWEL_MID,
        "œ": PhonemeCategory.VOWEL_MID,
        "ɤ": PhonemeCategory.VOWEL_MID,
        # Open vowels
        "a": PhonemeCategory.VOWEL_OPEN,
        "ɑ": PhonemeCategory.VOWEL_OPEN,
        "æ": PhonemeCategory.VOWEL_OPEN,
        "ɐ": PhonemeCategory.VOWEL_OPEN,
        # Plosives (stops)
        "p": PhonemeCategory.PLOSIVE,
        "b": PhonemeCategory.PLOSIVE,
        "t": PhonemeCategory.PLOSIVE,
        "d": PhonemeCategory.PLOSIVE,
        "k": PhonemeCategory.PLOSIVE,
        "g": PhonemeCategory.PLOSIVE,
        "ʔ": PhonemeCategory.PLOSIVE,  # Glottal stop
        # Fricatives (non-sibilant)
        "f": PhonemeCategory.FRICATIVE,
        "v": PhonemeCategory.FRICATIVE,
        "θ": PhonemeCategory.FRICATIVE,  # 'th' in 'think'
        "ð": PhonemeCategory.FRICATIVE,  # 'th' in 'this'
        "h": PhonemeCategory.FRICATIVE,
        "x": PhonemeCategory.FRICATIVE,  # German 'ch'
        "ɣ": PhonemeCategory.FRICATIVE,
        # Nasals
        "m": PhonemeCategory.NASAL,
        "n": PhonemeCategory.NASAL,
        "ŋ": PhonemeCategory.NASAL,  # 'ng' in 'sing'
        "ɲ": PhonemeCategory.NASAL,
        # Liquids
        "l": PhonemeCategory.LIQUID,
        "r": PhonemeCategory.LIQUID,
        "ɹ": PhonemeCategory.LIQUID,  # English 'r'
        "ɾ": PhonemeCategory.LIQUID,  # Spanish 'r'
        "ʀ": PhonemeCategory.LIQUID,  # Uvular 'r'
        # Glides (semivowels)
        "w": PhonemeCategory.GLIDE,
        "j": PhonemeCategory.GLIDE,  # 'y' in 'yes'
        "ɥ": PhonemeCategory.GLIDE,
        # Sibilants - Alveolar
        "s": PhonemeCategory.SIBILANT_ALVEOLAR,
        "z": PhonemeCategory.SIBILANT_ALVEOLAR,
        # Sibilants - Postalveolar
        "ʃ": PhonemeCategory.SIBILANT_POSTALVEOLAR,  # 'sh'
        "ʒ": PhonemeCategory.SIBILANT_POSTALVEOLAR,  # 's' in 'measure'
        # Sibilants - Affricates
        "tʃ": PhonemeCategory.SIBILANT_AFFRICATE,  # 'ch'
        "dʒ": PhonemeCategory.SIBILANT_AFFRICATE,  # 'j' in 'judge'
        "ʧ": PhonemeCategory.SIBILANT_AFFRICATE,  # Alternative 'ch'
        "ʤ": PhonemeCategory.SIBILANT_AFFRICATE,  # Alternative 'j'
    }

    # Sibilant detailed classification
    _SIBILANT_MAPPINGS: dict[str, SibilantType] = {
        "s": SibilantType.S_VOICELESS,
        "z": SibilantType.Z_VOICED,
        "ʃ": SibilantType.SH_VOICELESS,
        "ʒ": SibilantType.ZH_VOICED,
        "tʃ": SibilantType.CH_VOICELESS,
        "dʒ": SibilantType.JH_VOICED,
        "ʧ": SibilantType.CH_VOICELESS,
        "ʤ": SibilantType.JH_VOICED,
    }

    # Voicing (True = voiced, False = voiceless)
    _VOICED_PHONEMES: set[str] = {
        # Voiced vowels (all vowels are voiced)
        "i",
        "u",
        "e",
        "o",
        "a",
        "ɑ",
        "ɛ",
        "ɔ",
        "ə",
        "ɪ",
        "ʊ",
        "æ",
        "y",
        "ø",
        "œ",
        "ɨ",
        "ʉ",
        "ɯ",
        "ɐ",
        "ɤ",
        # Voiced consonants
        "b",
        "d",
        "g",
        "v",
        "ð",
        "z",
        "ʒ",
        "dʒ",
        "ʤ",
        "m",
        "n",
        "ŋ",
        "ɲ",
        "l",
        "r",
        "ɹ",
        "ɾ",
        "ʀ",
        "w",
        "j",
        "ɥ",
        "ɣ",
    }

    # Articulation place for consonants
    _PLACE_MAPPINGS: dict[str, ArticulationPlace] = {
        # Bilabial
        "p": ArticulationPlace.BILABIAL,
        "b": ArticulationPlace.BILABIAL,
        "m": ArticulationPlace.BILABIAL,
        "w": ArticulationPlace.BILABIAL,
        # Labiodental
        "f": ArticulationPlace.LABIODENTAL,
        "v": ArticulationPlace.LABIODENTAL,
        # Dental
        "θ": ArticulationPlace.DENTAL,
        "ð": ArticulationPlace.DENTAL,
        # Alveolar
        "t": ArticulationPlace.ALVEOLAR,
        "d": ArticulationPlace.ALVEOLAR,
        "s": ArticulationPlace.ALVEOLAR,
        "z": ArticulationPlace.ALVEOLAR,
        "n": ArticulationPlace.ALVEOLAR,
        "l": ArticulationPlace.ALVEOLAR,
        "ɹ": ArticulationPlace.ALVEOLAR,
        # Postalveolar
        "ʃ": ArticulationPlace.POSTALVEOLAR,
        "ʒ": ArticulationPlace.POSTALVEOLAR,
        "tʃ": ArticulationPlace.POSTALVEOLAR,
        "dʒ": ArticulationPlace.POSTALVEOLAR,
        "ʧ": ArticulationPlace.POSTALVEOLAR,
        "ʤ": ArticulationPlace.POSTALVEOLAR,
        # Palatal
        "j": ArticulationPlace.PALATAL,
        # Velar
        "k": ArticulationPlace.VELAR,
        "g": ArticulationPlace.VELAR,
        "ŋ": ArticulationPlace.VELAR,
        "x": ArticulationPlace.VELAR,
        # Glottal
        "h": ArticulationPlace.GLOTTAL,
        "ʔ": ArticulationPlace.GLOTTAL,
    }

    def __init__(self):
        """Initialisiert phoneme classifier."""
        logger.info("PhonemeClassifier initialisiert with %s phoneme mappings", len(self._PHONEME_MAPPINGS))

    def classify(self, phoneme: str) -> PhonemeCategory:
        """
        Classify phoneme into main category.

        Args:
            phoneme: IPA phoneme symbol

        Returns:
            Main phoneme category

        Example:
            >>> classifier = PhonemeClassifier()
            >>> classifier.classify('s')
            PhonemeCategory.SIBILANT_ALVEOLAR
        """
        phoneme = phoneme.strip().lower()
        return self._PHONEME_MAPPINGS.get(phoneme, PhonemeCategory.UNKNOWN)

    def classify_detailed(self, phoneme: str) -> PhonemeInfo:
        """
        Gibt zurück: detailed phoneme classification.

        Args:
            phoneme: IPA phoneme symbol

        Returns:
            Detailed phoneme information

        Example:
            >>> classifier = PhonemeClassifier()
            >>> info = classifier.classify_detailed('tʃ')
            >>> logger.info(f"Category: {info.category}")
            >>> logger.info(f"Sibilant: {info.is_sibilant}")
            >>> logger.info(f"Type: {info.sibilant_type}")
        """
        phoneme = phoneme.strip().lower()
        category = self.classify(phoneme)

        # Determine properties
        is_vowel = category in [PhonemeCategory.VOWEL_CLOSE, PhonemeCategory.VOWEL_MID, PhonemeCategory.VOWEL_OPEN]

        is_consonant = not is_vowel and category not in [
            PhonemeCategory.SILENCE,
            PhonemeCategory.BREATH,
            PhonemeCategory.UNKNOWN,
        ]

        is_sibilant = category in [
            PhonemeCategory.SIBILANT_ALVEOLAR,
            PhonemeCategory.SIBILANT_POSTALVEOLAR,
            PhonemeCategory.SIBILANT_AFFRICATE,
        ]

        is_voiced = phoneme in self._VOICED_PHONEMES

        sibilant_type = self._SIBILANT_MAPPINGS.get(phoneme) if is_sibilant else None

        place = self._PLACE_MAPPINGS.get(phoneme) if is_consonant else None

        return PhonemeInfo(
            phoneme=phoneme,
            category=category,
            is_vowel=is_vowel,
            is_consonant=is_consonant,
            is_sibilant=is_sibilant,
            is_voiced=is_voiced,
            sibilant_type=sibilant_type,
            place=place,
        )

    def is_vowel(self, phoneme: str) -> bool:
        """Prüft if phoneme is a vowel."""
        category = self.classify(phoneme)
        return category in [PhonemeCategory.VOWEL_CLOSE, PhonemeCategory.VOWEL_MID, PhonemeCategory.VOWEL_OPEN]

    def is_consonant(self, phoneme: str) -> bool:
        """Prüft if phoneme is a consonant."""
        return not self.is_vowel(phoneme) and self.classify(phoneme) not in [
            PhonemeCategory.SILENCE,
            PhonemeCategory.BREATH,
            PhonemeCategory.UNKNOWN,
        ]

    def is_sibilant(self, phoneme: str) -> bool:
        """Prüft if phoneme is a sibilant."""
        category = self.classify(phoneme)
        return category in [
            PhonemeCategory.SIBILANT_ALVEOLAR,
            PhonemeCategory.SIBILANT_POSTALVEOLAR,
            PhonemeCategory.SIBILANT_AFFRICATE,
        ]

    def is_voiced(self, phoneme: str) -> bool:
        """Prüft if phoneme is voiced."""
        phoneme = phoneme.strip().lower()
        return phoneme in self._VOICED_PHONEMES

    def get_sibilant_type(self, phoneme: str) -> SibilantType | None:
        """
        Gibt zurück: detailed sibilant type.

        Args:
            phoneme: IPA phoneme symbol

        Returns:
            Sibilant type if phoneme is a sibilant, else None
        """
        phoneme = phoneme.strip().lower()
        return self._SIBILANT_MAPPINGS.get(phoneme)

    def get_place(self, phoneme: str) -> ArticulationPlace | None:
        """
        Gibt zurück: articulation place for consonant.

        Args:
            phoneme: IPA phoneme symbol

        Returns:
            Articulation place if phoneme is a consonant, else None
        """
        phoneme = phoneme.strip().lower()
        return self._PLACE_MAPPINGS.get(phoneme)

    def get_frequency_center(self, phoneme: str) -> float | None:
        """
        Gibt zurück: typical spectral center frequency for sibilant.

        Args:
            phoneme: IPA phoneme symbol

        Returns:
            Center frequency in Hz if sibilant, else None

        Example:
            >>> classifier = PhonemeClassifier()
            >>> freq = classifier.get_frequency_center('s')
            >>> logger.info(f"Frequency center: {freq} Hz")
            8000.0 Hz
        """
        sibilant_type = self.get_sibilant_type(phoneme)

        if sibilant_type is None:
            return None

        # Typical center frequencies for sibilants
        frequency_map = {
            SibilantType.S_VOICELESS: 8000.0,
            SibilantType.Z_VOICED: 7500.0,
            SibilantType.SH_VOICELESS: 5000.0,
            SibilantType.ZH_VOICED: 4500.0,
            SibilantType.CH_VOICELESS: 6000.0,
            SibilantType.JH_VOICED: 5500.0,
        }

        return frequency_map.get(sibilant_type)

    def get_supported_phonemes(self) -> set[str]:
        """Gibt zurück: set of all supported IPA phonemes."""
        return set(self._PHONEME_MAPPINGS.keys())

    def get_statistics(self) -> dict[str, int]:
        """
        Gibt zurück: classifier statistics.

        Returns:
            Dictionary with counts:
            - total_phonemes: Total supported phonemes
            - vowels: Number of vowels
            - consonants: Number of consonants
            - sibilants: Number of sibilants
            - voiced: Number of voiced phonemes
        """
        all_phonemes = self.get_supported_phonemes()

        vowels = sum(1 for p in all_phonemes if self.is_vowel(p))
        consonants = sum(1 for p in all_phonemes if self.is_consonant(p))
        sibilants = sum(1 for p in all_phonemes if self.is_sibilant(p))
        voiced = sum(1 for p in all_phonemes if self.is_voiced(p))

        return {
            "total_phonemes": len(all_phonemes),
            "vowels": vowels,
            "consonants": consonants,
            "sibilants": sibilants,
            "voiced": voiced,
        }
