import pytest

"""
Unit tests for backend/core/dsp/silence_mask.py

Normative Anforderungen:
  §silence-guarantee: gewollte Stille bleibt still — nach jeder Phase garantiert.
  §2.46f: Atemgeräusche (spectral_flatness > 0.4, RMS in [-57, -38] dBFS) sind
          KEINE Stille — sie dürfen nicht als Stilleregion markiert werden.
  §0h   : Kein Eingriff darf Stille in Musik verwandeln.
"""

import gc

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _music(duration_s: float = 2.0, sr: int = 48_000, level: float = 0.3) -> np.ndarray:
    """Synthetisches Music-Signal (Sinus-Summe) auf gegebenem Pegel."""
    t = np.linspace(0.0, duration_s, int(duration_s * sr), endpoint=False)
    sig = (np.sin(2 * np.pi * 440.0 * t) + 0.5 * np.sin(2 * np.pi * 880.0 * t)) * level
    return sig.astype(np.float32)  # type: ignore[no-any-return]


def _silence(duration_s: float = 1.0, sr: int = 48_000) -> np.ndarray:
    """Echte Stille (Nullen)."""
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _breath(duration_s: float = 0.3, sr: int = 48_000, level_dbfs: float = -46.0) -> np.ndarray:
    """Synthetisches Atemgeräusch: breitbandiges Rauschen bei geringer Lautstärke.

    Normiert auf RMS-Pegel, da Atemgeräusch-Schutz RMS-basiert prüft.
    """
    n = int(duration_s * sr)
    sig = np.random.default_rng(42).standard_normal(n).astype(np.float32)
    rms = float(np.sqrt(np.mean(sig.astype(np.float64) ** 2)))
    if rms > 1e-12:
        sig /= rms  # auf RMS=1 normieren
    target_rms = 10.0 ** (level_dbfs / 20.0)
    sig *= target_rms
    return sig


# ---------------------------------------------------------------------------
# Tests: Modul-Import
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModuleImport:
    def test_module_imports_cleanly(self):
        """silence_mask.py kann fehlerfrei importiert werden."""
        from backend.core.dsp.silence_mask import (
            apply_silence_preservation,
            compute_silence_mask,
            get_silence_mask_computer,
        )

        assert callable(compute_silence_mask)
        assert callable(apply_silence_preservation)
        assert callable(get_silence_mask_computer)

    def test_material_silence_db_table_present(self):
        """Alle kritischen Materialtypen haben einen Stille-Schwellwert."""
        from backend.core.dsp.silence_mask import _MATERIAL_SILENCE_DB

        for mat in ("shellac", "vinyl", "cassette", "cd_digital", "unknown"):
            assert mat in _MATERIAL_SILENCE_DB, f"{mat} fehlt in _MATERIAL_SILENCE_DB"

    def test_shellac_threshold_coarser_than_cd(self):
        """Shellac-Schwellwert muss gröber sein als CD-Schwellwert (Rauschen-Artefakt vs. digital)."""
        from backend.core.dsp.silence_mask import _MATERIAL_SILENCE_DB

        assert _MATERIAL_SILENCE_DB["shellac"] > _MATERIAL_SILENCE_DB["cd_digital"]

    def teardown_method(self):
        gc.collect(0)


# ---------------------------------------------------------------------------
# Tests: Masken-Berechnung
# ---------------------------------------------------------------------------


class TestComputeSilenceMask:
    SR = 48_000

    def test_digital_silence_detected(self):
        """`compute_silence_mask` erkennt echte Stille (Null-Signal) korrekt."""
        from backend.core.dsp.silence_mask import compute_silence_mask

        audio = np.concatenate([_music(2.0, self.SR), _silence(1.0, self.SR)])
        mask = compute_silence_mask(audio, self.SR, material_key="cd_digital")

        assert mask.shape == (len(audio),)
        assert mask.dtype == np.float32

        # Stille-Zone: mask nahe 0
        sil_core = mask[2 * self.SR + self.SR // 4 : 3 * self.SR - self.SR // 4]
        assert float(np.mean(sil_core)) < 0.15, "Stille nicht als solche erkannt"

    def test_music_zone_not_muted(self):
        """`compute_silence_mask` markiert Musik-Passagen NICHT als Stille."""
        from backend.core.dsp.silence_mask import compute_silence_mask

        audio = _music(2.0, self.SR, level=0.3)
        mask = compute_silence_mask(audio, self.SR, material_key="vinyl")

        music_core = mask[self.SR // 2 : 3 * self.SR // 2]
        assert float(np.mean(music_core)) > 0.8, "Musik fälschlicherweise als Stille markiert"

    def test_vinyl_level_silence_detected_with_vinyl_key(self):
        """Vinyl-Rauschen (~-45 dBFS) mit vinyl-Schlüssel → als Stille erkannt."""
        from backend.core.dsp.silence_mask import compute_silence_mask

        # Stille auf Vinyl-Pegel: weißes Rauschen bei ~-45 dBFS
        rng = np.random.default_rng(0)
        vinyl_noise = rng.standard_normal(2 * self.SR).astype(np.float32)
        vinyl_noise *= 10.0 ** (-45.0 / 20.0)
        music_seg = _music(1.0, self.SR, level=0.3)
        audio = np.concatenate([music_seg, vinyl_noise])

        mask = compute_silence_mask(audio, self.SR, material_key="vinyl")
        vinyl_core = mask[self.SR + self.SR // 4 : 3 * self.SR - self.SR // 4]
        assert float(np.mean(vinyl_core)) < 0.20, "Vinyl-Stille nicht erkannt"

    def test_breath_region_not_marked_as_silence(self):
        """§2.46f: Atemgeräusch (breitband, −46 dBFS) darf NICHT als Stille markiert werden."""
        from backend.core.dsp.silence_mask import compute_silence_mask

        music_seg = _music(1.0, self.SR, level=0.3)
        breath_seg = _breath(0.4, self.SR, level_dbfs=-46.0)
        audio = np.concatenate([music_seg, breath_seg])

        mask = compute_silence_mask(audio, self.SR, material_key="cd_digital", protect_breath=True)

        # Atemzone soll NICHT als Stille gelten (mask > 0.5 im Kern)
        breath_start = len(music_seg)
        breath_core = mask[breath_start + self.SR // 20 : breath_start + len(breath_seg) - self.SR // 20]
        # Atemsegment ist kurz — mind. 50% der Frames dürfen nicht als Stille markiert sein
        breath_active_frac = float(np.mean(breath_core > 0.5))
        assert breath_active_frac >= 0.5, (
            f"§2.46f VERLETZT: Atemgeräusch als Stille markiert (aktiv-Anteil={breath_active_frac:.2f})"
        )

    def test_crossfade_smooth_ramps(self):
        """Maske hat glatte Übergänge (crossfade) an Stille-Grenzen — kein Sprung."""
        from backend.core.dsp.silence_mask import compute_silence_mask

        audio = np.concatenate([_music(1.0, self.SR), _silence(1.0, self.SR)])
        mask = compute_silence_mask(audio, self.SR, material_key="cd_digital", crossfade_s=0.020)

        # Suche Übergangsbereich: Maske sinkt von 1 → 0 nach Ende der Musik
        transition_zone = mask[self.SR - 1500 : self.SR + 1500]
        # Kein harter Sprung: max. Änderung von Sample zu Sample < 0.1
        max_step = float(np.max(np.abs(np.diff(transition_zone.astype(np.float64)))))
        assert max_step < 0.1, f"Zu großer Schritt an Stille-Grenze: {max_step:.4f} (Crossfade fehlt?)"

    def test_stereo_audio_handled(self):
        """Stereo-Audio (2, N) wird ohne Fehler verarbeitet."""
        from backend.core.dsp.silence_mask import compute_silence_mask

        mono = np.concatenate([_music(1.0, self.SR), _silence(0.5, self.SR)])
        stereo = np.stack([mono, mono], axis=0)  # (2, N)
        mask = compute_silence_mask(stereo, self.SR, material_key="vinyl")
        assert mask.ndim == 1  # mask ist immer 1D
        assert mask.shape[0] == stereo.shape[-1]

    def test_returns_float32(self):
        """Rückgabe MUSS float32 sein (Kompatibilität mit pipeline-Arithmetik)."""
        from backend.core.dsp.silence_mask import compute_silence_mask

        audio = _music(0.5, self.SR)
        mask = compute_silence_mask(audio, self.SR, material_key="unknown")
        assert mask.dtype == np.float32

    def teardown_method(self):
        gc.collect(0)


# ---------------------------------------------------------------------------
# Tests: apply_silence_preservation
# ---------------------------------------------------------------------------


class TestApplySilencePreservation:
    SR = 48_000

    def test_silence_zone_perfectly_restored(self):
        """Stille-Zone wird exakt wiederhergestellt — kein Restpegel."""
        from backend.core.dsp.silence_mask import apply_silence_preservation, compute_silence_mask

        original = np.concatenate([_music(2.0, self.SR), _silence(1.0, self.SR)])
        mask = compute_silence_mask(original, self.SR, material_key="cd_digital")

        # Simuliere Pegelexplosion: Inpainting füllt Stille mit Musik-Level-Inhalt
        processed = original.copy()
        processed[2 * self.SR :] = 0.4  # inject loud content into silence

        result = apply_silence_preservation(original, processed, mask)

        # In der tiefen Stille-Zone (gut weg von den Rändern) muss das Ergebnis
        # nahezu identisch mit dem Original (≈ 0.0) sein.
        sil_core = result[2 * self.SR + self.SR // 4 : 3 * self.SR - self.SR // 4]
        assert float(np.max(np.abs(sil_core))) < 0.01, (
            f"§silence-guarantee VERLETZT: Stille-Zone mit Level {float(np.max(np.abs(sil_core))):.4f} != 0"
        )

    def test_music_zone_unchanged(self):
        """Musik-Zone darf durch apply_silence_preservation NICHT verändert werden."""
        from backend.core.dsp.silence_mask import apply_silence_preservation, compute_silence_mask

        original = np.concatenate([_music(2.0, self.SR), _silence(1.0, self.SR)])
        mask = compute_silence_mask(original, self.SR, material_key="cd_digital")

        processed = original * 0.95  # leichte Dämpfung durch Phase simuliert
        result = apply_silence_preservation(original, processed, mask)

        music_core = result[self.SR // 4 : 7 * self.SR // 4]
        expected = processed[self.SR // 4 : 7 * self.SR // 4]
        max_diff = float(np.max(np.abs(music_core - expected)))
        assert max_diff < 0.01, f"Musik-Zone durch Stille-Maske verändert: max_diff={max_diff:.6f}"

    def test_stereo_preservation(self):
        """Stereo-Audio (2, N) wird korrekt behandelt — beide Kanäle geschützt."""
        from backend.core.dsp.silence_mask import apply_silence_preservation, compute_silence_mask

        mono_orig = np.concatenate([_music(1.0, self.SR), _silence(0.5, self.SR)])
        stereo_orig = np.stack([mono_orig, mono_orig * 0.9], axis=0)  # (2, N)
        mask = compute_silence_mask(mono_orig, self.SR, material_key="vinyl")

        # Inject content in silence zone
        stereo_proc = stereo_orig.copy()
        stereo_proc[:, self.SR :] = 0.3

        result = apply_silence_preservation(stereo_orig, stereo_proc, mask)
        assert result.shape == stereo_orig.shape

        # Stille-Kern in beiden Kanälen erhalten
        sil_core = result[:, self.SR + self.SR // 8 : 3 * self.SR // 2 - self.SR // 8]
        assert float(np.max(np.abs(sil_core))) < 0.01

    def test_all_active_mask_passthrough(self):
        """Maske mit allen Einsen → Ergebnis == processed (keine Änderung)."""
        from backend.core.dsp.silence_mask import apply_silence_preservation

        original = _music(0.5, self.SR)
        processed = original * 0.8
        mask_ones = np.ones(len(original), dtype=np.float32)

        result = apply_silence_preservation(original, processed, mask_ones)
        max_diff = float(np.max(np.abs(result - processed)))
        assert max_diff < 1e-6

    def test_all_silent_mask_returns_original(self):
        """Maske mit allen Nullen → Ergebnis == original."""
        from backend.core.dsp.silence_mask import apply_silence_preservation

        original = _music(0.5, self.SR)
        processed = original * 0.8
        mask_zeros = np.zeros(len(original), dtype=np.float32)

        result = apply_silence_preservation(original, processed, mask_zeros)
        max_diff = float(np.max(np.abs(result - original)))
        assert max_diff < 1e-6

    def teardown_method(self):
        gc.collect(0)


# ---------------------------------------------------------------------------
# Tests: Pegelexplosions-Regression
# ---------------------------------------------------------------------------


class TestPegelexplosionRegression:
    """Regression: gewollte Stille am Song-Anfang/-Ende/-Mitte darf nicht verstärkt werden."""

    SR = 48_000

    def _make_song_with_silence(self) -> np.ndarray:
        """Intro-Stille + Musik + Outro-Stille."""
        return np.concatenate(  # type: ignore[no-any-return]
            [
                _silence(1.0, self.SR),  # Intro
                _music(3.0, self.SR, level=0.3),
                _silence(1.0, self.SR),  # Outro
            ]
        ).astype(np.float32)

    def test_silence_not_amplified_after_phase_simulation(self):
        """Nach simulierter Phase, die Stille mit Music-Level füllt: Stille bleibt still."""
        from backend.core.dsp.silence_mask import apply_silence_preservation, compute_silence_mask

        original = self._make_song_with_silence()
        mask = compute_silence_mask(original, self.SR, material_key="cd_digital")

        # Phase füllt Intro- und Outro-Stille komplett mit Inhalt (Pegelexplosion!)
        processed = original.copy()
        processed[: self.SR] = 0.35  # Intro: 0 → 0.35 (Pegelexplosion)
        processed[4 * self.SR :] = 0.35  # Outro: 0 → 0.35 (Pegelexplosion)

        result = apply_silence_preservation(original, processed, mask)

        # Intro-Kern: Gut weg von den Rändern — muss nahezu Stille bleiben
        intro_core = result[self.SR // 4 : 3 * self.SR // 4]
        assert float(np.max(np.abs(intro_core))) < 0.02, (
            f"§silence-guarantee VERLETZT: Intro-Pegelexplosion nicht verhindert "
            f"(max={float(np.max(np.abs(intro_core))):.4f})"
        )

        # Outro-Kern: ebenso
        outro_core = result[4 * self.SR + self.SR // 4 : 5 * self.SR - self.SR // 4]
        assert float(np.max(np.abs(outro_core))) < 0.02, (
            f"§silence-guarantee VERLETZT: Outro-Pegelexplosion nicht verhindert "
            f"(max={float(np.max(np.abs(outro_core))):.4f})"
        )

    def test_interior_silence_protected(self):
        """Interne Stille (mitten im Song) bleibt nach Phase-Simulation still."""
        from backend.core.dsp.silence_mask import apply_silence_preservation, compute_silence_mask

        # Musik — Interne Stille — Musik
        original = np.concatenate([_music(1.5, self.SR), _silence(0.5, self.SR), _music(1.5, self.SR)]).astype(
            np.float32
        )
        mask = compute_silence_mask(original, self.SR, material_key="vinyl")

        processed = original.copy()
        # Phase füllt interne Stille mit AR-Interpolation (Pegelexplosion)
        processed[int(1.5 * self.SR) : 2 * self.SR] = 0.3

        result = apply_silence_preservation(original, processed, mask)

        sil_core_start = int(1.5 * self.SR) + self.SR // 8
        sil_core_end = 2 * self.SR - self.SR // 8
        sil_core = result[sil_core_start:sil_core_end]
        assert float(np.max(np.abs(sil_core))) < 0.02, (
            f"§silence-guarantee VERLETZT: Interne Stille nach Phase verstärkt "
            f"(max={float(np.max(np.abs(sil_core))):.4f})"
        )

    def teardown_method(self):
        gc.collect(0)


# ---------------------------------------------------------------------------
# Tests: Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_singleton_returns_same_instance(self):
        """get_silence_mask_computer() MUSS immer dieselbe Instanz zurückgeben."""
        from backend.core.dsp.silence_mask import get_silence_mask_computer

        inst1 = get_silence_mask_computer()
        inst2 = get_silence_mask_computer()
        assert inst1 is inst2

    def teardown_method(self):
        gc.collect(0)
