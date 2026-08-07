"""
Tests for PhonemeClassifier - IPA phoneme classification

Tests the phoneme classification logic including:
- Category classification (vowels, consonants, sibilants)
- Sibilant type detection
- Voicing detection
- Articulation place detection
- Frequency center estimation

Note: These tests require NO external dependencies (no torch/transformers).
All classification is done via lookup tables.

Author: Aurik Development Team
Version: 1.0.0
"""

import pytest

from backend.ml.phoneme_aware import PhonemeCategory, PhonemeClassifier, SibilantType
from backend.ml.phoneme_aware.phoneme_classifier import ArticulationPlace, PhonemeInfo


@pytest.mark.unit
class TestPhonemeClassifierBasic:
    """Test basic classification functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = PhonemeClassifier()

    def test_initialization(self):
        """Test classifier initialization"""
        assert self.classifier is not None
        assert isinstance(self.classifier, PhonemeClassifier)

    def test_classify_vowels(self):
        """Test vowel classification"""
        # Close vowels
        assert self.classifier.classify("i") == PhonemeCategory.VOWEL_CLOSE
        assert self.classifier.classify("u") == PhonemeCategory.VOWEL_CLOSE
        assert self.classifier.classify("y") == PhonemeCategory.VOWEL_CLOSE

        # Mid vowels
        assert self.classifier.classify("e") == PhonemeCategory.VOWEL_MID
        assert self.classifier.classify("o") == PhonemeCategory.VOWEL_MID
        assert self.classifier.classify("ə") == PhonemeCategory.VOWEL_MID

        # Open vowels
        assert self.classifier.classify("a") == PhonemeCategory.VOWEL_OPEN
        assert self.classifier.classify("æ") == PhonemeCategory.VOWEL_OPEN
        assert self.classifier.classify("ɑ") == PhonemeCategory.VOWEL_OPEN

    def test_classify_consonants(self):
        """Test consonant classification"""
        # Plosives
        assert self.classifier.classify("p") == PhonemeCategory.PLOSIVE
        assert self.classifier.classify("t") == PhonemeCategory.PLOSIVE
        assert self.classifier.classify("k") == PhonemeCategory.PLOSIVE
        assert self.classifier.classify("b") == PhonemeCategory.PLOSIVE
        assert self.classifier.classify("d") == PhonemeCategory.PLOSIVE
        assert self.classifier.classify("g") == PhonemeCategory.PLOSIVE

        # Fricatives (non-sibilant)
        assert self.classifier.classify("f") == PhonemeCategory.FRICATIVE
        assert self.classifier.classify("v") == PhonemeCategory.FRICATIVE
        assert self.classifier.classify("θ") == PhonemeCategory.FRICATIVE
        assert self.classifier.classify("ð") == PhonemeCategory.FRICATIVE

        # Nasals
        assert self.classifier.classify("m") == PhonemeCategory.NASAL
        assert self.classifier.classify("n") == PhonemeCategory.NASAL
        assert self.classifier.classify("ŋ") == PhonemeCategory.NASAL

        # Liquids
        assert self.classifier.classify("l") == PhonemeCategory.LIQUID
        assert self.classifier.classify("r") == PhonemeCategory.LIQUID
        assert self.classifier.classify("ɹ") == PhonemeCategory.LIQUID

    def test_classify_sibilants(self):
        """Test sibilant classification"""
        # Alveolar sibilants
        assert self.classifier.classify("s") == PhonemeCategory.SIBILANT_ALVEOLAR
        assert self.classifier.classify("z") == PhonemeCategory.SIBILANT_ALVEOLAR

        # Postalveolar sibilants
        assert self.classifier.classify("ʃ") == PhonemeCategory.SIBILANT_POSTALVEOLAR
        assert self.classifier.classify("ʒ") == PhonemeCategory.SIBILANT_POSTALVEOLAR

        # Affricate sibilants
        assert self.classifier.classify("tʃ") == PhonemeCategory.SIBILANT_AFFRICATE
        assert self.classifier.classify("dʒ") == PhonemeCategory.SIBILANT_AFFRICATE

    def test_classify_unknown(self):
        """Test classification of unknown phonemes"""
        assert self.classifier.classify("xyz") == PhonemeCategory.UNKNOWN
        assert self.classifier.classify("12345") == PhonemeCategory.UNKNOWN
        assert self.classifier.classify("") == PhonemeCategory.UNKNOWN


class TestPhonemeClassifierDetailed:
    """Test detailed classification with PhonemeInfo"""

    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = PhonemeClassifier()

    def test_classify_detailed_vowel(self):
        """Test detailed vowel classification"""
        info = self.classifier.classify_detailed("a")

        assert isinstance(info, PhonemeInfo)
        assert info.phoneme == "a"
        assert info.category == PhonemeCategory.VOWEL_OPEN
        assert info.is_vowel is True
        assert info.is_consonant is False
        assert info.is_sibilant is False
        assert info.is_voiced is True
        assert info.sibilant_type is None
        assert info.place is None

    def test_classify_detailed_consonant(self):
        """Test detailed consonant classification"""
        info = self.classifier.classify_detailed("p")

        assert info.phoneme == "p"
        assert info.category == PhonemeCategory.PLOSIVE
        assert info.is_vowel is False
        assert info.is_consonant is True
        assert info.is_sibilant is False
        assert info.is_voiced is False
        assert info.sibilant_type is None
        assert info.place == ArticulationPlace.BILABIAL

    def test_classify_detailed_sibilant_voiceless(self):
        """Test detailed sibilant classification (voiceless)"""
        info = self.classifier.classify_detailed("s")

        assert info.phoneme == "s"
        assert info.category == PhonemeCategory.SIBILANT_ALVEOLAR
        assert info.is_vowel is False
        assert info.is_consonant is True
        assert info.is_sibilant is True
        assert info.is_voiced is False
        assert info.sibilant_type == SibilantType.S_VOICELESS
        assert info.place == ArticulationPlace.ALVEOLAR

    def test_classify_detailed_sibilant_voiced(self):
        """Test detailed sibilant classification (voiced)"""
        info = self.classifier.classify_detailed("z")

        assert info.phoneme == "z"
        assert info.category == PhonemeCategory.SIBILANT_ALVEOLAR
        assert info.is_sibilant is True
        assert info.is_voiced is True
        assert info.sibilant_type == SibilantType.Z_VOICED

    def test_classify_detailed_all_sibilants(self):
        """Test all sibilant types"""
        sibilants = {
            "s": (SibilantType.S_VOICELESS, False, PhonemeCategory.SIBILANT_ALVEOLAR),
            "z": (SibilantType.Z_VOICED, True, PhonemeCategory.SIBILANT_ALVEOLAR),
            "ʃ": (SibilantType.SH_VOICELESS, False, PhonemeCategory.SIBILANT_POSTALVEOLAR),
            "ʒ": (SibilantType.ZH_VOICED, True, PhonemeCategory.SIBILANT_POSTALVEOLAR),
            "tʃ": (SibilantType.CH_VOICELESS, False, PhonemeCategory.SIBILANT_AFFRICATE),
            "dʒ": (SibilantType.JH_VOICED, True, PhonemeCategory.SIBILANT_AFFRICATE),
        }

        for phoneme, (sib_type, voiced, category) in sibilants.items():
            info = self.classifier.classify_detailed(phoneme)
            assert info.is_sibilant is True, f"/{phoneme}/ should be sibilant"
            assert info.sibilant_type == sib_type, f"/{phoneme}/ type mismatch"
            assert info.is_voiced is voiced, f"/{phoneme}/ voicing mismatch"
            assert info.category == category, f"/{phoneme}/ category mismatch"


class TestPhonemeClassifierBooleanChecks:
    """Test boolean helper methods"""

    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = PhonemeClassifier()

    def test_is_vowel(self):
        """Test is_vowel() method"""
        # Vowels
        assert self.classifier.is_vowel("a") is True
        assert self.classifier.is_vowel("e") is True
        assert self.classifier.is_vowel("i") is True
        assert self.classifier.is_vowel("o") is True
        assert self.classifier.is_vowel("u") is True

        # Non-vowels
        assert self.classifier.is_vowel("p") is False
        assert self.classifier.is_vowel("s") is False
        assert self.classifier.is_vowel("m") is False

    def test_is_consonant(self):
        """Test is_consonant() method"""
        # Consonants
        assert self.classifier.is_consonant("p") is True
        assert self.classifier.is_consonant("t") is True
        assert self.classifier.is_consonant("s") is True
        assert self.classifier.is_consonant("m") is True

        # Non-consonants
        assert self.classifier.is_consonant("a") is False
        assert self.classifier.is_consonant("i") is False

    def test_is_sibilant(self):
        """Test is_sibilant() method"""
        # Sibilants
        assert self.classifier.is_sibilant("s") is True
        assert self.classifier.is_sibilant("z") is True
        assert self.classifier.is_sibilant("ʃ") is True
        assert self.classifier.is_sibilant("ʒ") is True
        assert self.classifier.is_sibilant("tʃ") is True
        assert self.classifier.is_sibilant("dʒ") is True

        # Non-sibilants
        assert self.classifier.is_sibilant("a") is False
        assert self.classifier.is_sibilant("p") is False
        assert self.classifier.is_sibilant("f") is False
        assert self.classifier.is_sibilant("m") is False

    def test_is_voiced(self):
        """Test is_voiced() method"""
        # Voiced
        assert self.classifier.is_voiced("a") is True  # Vowels are voiced
        assert self.classifier.is_voiced("b") is True
        assert self.classifier.is_voiced("d") is True
        assert self.classifier.is_voiced("z") is True
        assert self.classifier.is_voiced("m") is True
        assert self.classifier.is_voiced("n") is True

        # Voiceless
        assert self.classifier.is_voiced("p") is False
        assert self.classifier.is_voiced("t") is False
        assert self.classifier.is_voiced("s") is False
        assert self.classifier.is_voiced("f") is False


class TestPhonemeClassifierSibilants:
    """Test sibilant-specific functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = PhonemeClassifier()

    def test_get_sibilant_type(self):
        """Test get_sibilant_type() method"""
        assert self.classifier.get_sibilant_type("s") == SibilantType.S_VOICELESS
        assert self.classifier.get_sibilant_type("z") == SibilantType.Z_VOICED
        assert self.classifier.get_sibilant_type("ʃ") == SibilantType.SH_VOICELESS
        assert self.classifier.get_sibilant_type("ʒ") == SibilantType.ZH_VOICED
        assert self.classifier.get_sibilant_type("tʃ") == SibilantType.CH_VOICELESS
        assert self.classifier.get_sibilant_type("dʒ") == SibilantType.JH_VOICED

        # Non-sibilants
        assert self.classifier.get_sibilant_type("a") is None
        assert self.classifier.get_sibilant_type("p") is None

    def test_get_frequency_center(self):
        """Test get_frequency_center() method"""
        # Test all sibilant frequencies
        assert self.classifier.get_frequency_center("s") == 8000.0
        assert self.classifier.get_frequency_center("z") == 7500.0
        assert self.classifier.get_frequency_center("ʃ") == 5000.0
        assert self.classifier.get_frequency_center("ʒ") == 4500.0
        assert self.classifier.get_frequency_center("tʃ") == 6000.0
        assert self.classifier.get_frequency_center("dʒ") == 5500.0

        # Non-sibilants should return None
        assert self.classifier.get_frequency_center("a") is None
        assert self.classifier.get_frequency_center("p") is None

    def test_frequency_ordering(self):
        """Test that sibilant frequencies are ordered correctly"""
        # Voiceless should be higher than voiced counterparts
        assert self.classifier.get_frequency_center("s") > self.classifier.get_frequency_center("z")  # type: ignore[operator]
        assert self.classifier.get_frequency_center("ʃ") > self.classifier.get_frequency_center("ʒ")  # type: ignore[operator]
        assert self.classifier.get_frequency_center("tʃ") > self.classifier.get_frequency_center("dʒ")  # type: ignore[operator]

        # Alveolar should be higher than postalveolar
        assert self.classifier.get_frequency_center("s") > self.classifier.get_frequency_center("ʃ")  # type: ignore[operator]
        assert self.classifier.get_frequency_center("z") > self.classifier.get_frequency_center("ʒ")  # type: ignore[operator]


class TestPhonemeClassifierArticulation:
    """Test articulation place detection"""

    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = PhonemeClassifier()

    def test_get_place_bilabial(self):
        """Test bilabial consonants"""
        assert self.classifier.get_place("p") == ArticulationPlace.BILABIAL
        assert self.classifier.get_place("b") == ArticulationPlace.BILABIAL
        assert self.classifier.get_place("m") == ArticulationPlace.BILABIAL

    def test_get_place_alveolar(self):
        """Test alveolar consonants"""
        assert self.classifier.get_place("t") == ArticulationPlace.ALVEOLAR
        assert self.classifier.get_place("d") == ArticulationPlace.ALVEOLAR
        assert self.classifier.get_place("n") == ArticulationPlace.ALVEOLAR
        assert self.classifier.get_place("s") == ArticulationPlace.ALVEOLAR
        assert self.classifier.get_place("z") == ArticulationPlace.ALVEOLAR

    def test_get_place_velar(self):
        """Test velar consonants"""
        assert self.classifier.get_place("k") == ArticulationPlace.VELAR
        assert self.classifier.get_place("g") == ArticulationPlace.VELAR
        assert self.classifier.get_place("ŋ") == ArticulationPlace.VELAR

    def test_get_place_vowels(self):
        """Test that vowels return None for place"""
        assert self.classifier.get_place("a") is None
        assert self.classifier.get_place("i") is None
        assert self.classifier.get_place("u") is None


class TestPhonemeClassifierStatistics:
    """Test statistics and metadata methods"""

    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = PhonemeClassifier()

    def test_get_supported_phonemes(self):
        """Test get_supported_phonemes() method"""
        phonemes = self.classifier.get_supported_phonemes()

        assert isinstance(phonemes, set)
        assert len(phonemes) > 50  # Should have 80+ phonemes

        # Check some common phonemes are included
        assert "a" in phonemes
        assert "i" in phonemes
        assert "s" in phonemes
        assert "p" in phonemes
        assert "m" in phonemes

    def test_get_statistics(self):
        """Test get_statistics() method"""
        stats = self.classifier.get_statistics()

        assert isinstance(stats, dict)
        assert "total_phonemes" in stats
        assert "vowels" in stats
        assert "consonants" in stats
        assert "sibilants" in stats
        assert "voiced" in stats

        # Validate counts
        assert stats["total_phonemes"] > 50
        assert stats["vowels"] > 0
        assert stats["consonants"] > 0
        assert stats["sibilants"] >= 6  # At least 6 sibilants
        assert stats["voiced"] > 0


class TestPhonemeClassifierEdgeCases:
    """Test edge cases and error handling"""

    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = PhonemeClassifier()

    def test_empty_string(self):
        """Test classification of empty string"""
        assert self.classifier.classify("") == PhonemeCategory.UNKNOWN
        assert self.classifier.is_vowel("") is False
        assert self.classifier.is_consonant("") is False
        assert self.classifier.is_sibilant("") is False

    def test_whitespace(self):
        """Test classification of whitespace"""
        assert self.classifier.classify(" ") == PhonemeCategory.UNKNOWN
        assert self.classifier.classify("\t") == PhonemeCategory.UNKNOWN
        assert self.classifier.classify("\n") == PhonemeCategory.UNKNOWN

    def test_numbers_and_special_chars(self):
        """Test classification of numbers and special characters"""
        assert self.classifier.classify("123") == PhonemeCategory.UNKNOWN
        assert self.classifier.classify("!@#") == PhonemeCategory.UNKNOWN
        assert self.classifier.classify("...") == PhonemeCategory.UNKNOWN

    def test_case_sensitivity(self):
        """Test that phonemes are case-sensitive (IPA is case-sensitive)"""
        # IPA uses different symbols, but test if we have any lowercase/uppercase
        # Most IPA is lowercase, but some distinct symbols exist
        info_a = self.classifier.classify_detailed("a")
        assert info_a.category == PhonemeCategory.VOWEL_OPEN

    def test_multi_char_phonemes(self):
        """Test multi-character phoneme symbols"""
        # Affricates are multi-char
        assert self.classifier.classify("tʃ") == PhonemeCategory.SIBILANT_AFFRICATE
        assert self.classifier.classify("dʒ") == PhonemeCategory.SIBILANT_AFFRICATE
        assert self.classifier.is_sibilant("tʃ") is True
        assert self.classifier.is_sibilant("dʒ") is True


class TestPhonemeClassifierRealWorld:
    """Test real-world usage scenarios"""

    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = PhonemeClassifier()

    def test_classify_phoneme_sequence(self):
        """Test classification of a phoneme sequence"""
        # Simulate "hello" /hɛlo/
        phonemes = ["h", "ɛ", "l", "o"]

        categories = [self.classifier.classify(p) for p in phonemes]

        assert categories[0] == PhonemeCategory.FRICATIVE  # h
        assert categories[1] == PhonemeCategory.VOWEL_MID  # ɛ
        assert categories[2] == PhonemeCategory.LIQUID  # l
        assert categories[3] == PhonemeCategory.VOWEL_MID  # o

    def test_sibilant_detection_in_speech(self):
        """Test sibilant detection in speech sequence"""
        # Simulate "speech" /spiːtʃ/
        phonemes = ["s", "p", "i", "tʃ"]

        sibilants = [p for p in phonemes if self.classifier.is_sibilant(p)]

        assert len(sibilants) == 2
        assert "s" in sibilants
        assert "tʃ" in sibilants

        # Get sibilant frequencies for de-essing
        frequencies = {p: self.classifier.get_frequency_center(p) for p in sibilants}

        assert frequencies["s"] == 8000.0
        assert frequencies["tʃ"] == 6000.0

    def test_vowel_consonant_ratio(self):
        """Test computing vowel/consonant ratio"""
        # Simulate "audio" /ɔːdiəʊ/
        phonemes = ["ɔ", "d", "i", "ə", "ʊ"]

        vowels = sum(1 for p in phonemes if self.classifier.is_vowel(p))
        consonants = sum(1 for p in phonemes if self.classifier.is_consonant(p))

        assert vowels == 4  # ɔ, i, ə, ʊ
        assert consonants == 1  # d

        cv_ratio = consonants / vowels if vowels > 0 else 0
        assert 0 < cv_ratio < 1  # More vowels than consonants in "audio"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
