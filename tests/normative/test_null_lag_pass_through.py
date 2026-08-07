"""
§v10.14 Null-Lag Pass-Through: Integrationstest für STCG auf lag-freiem Material.

Ein kommerzielles Stereo-MP3 hat KEINEN inter-channel Lag. Jede STCG-Korrektur
auf solchem Material ist ein False Positive (Cross-Correlation fehlinterpretiert
MP3 Joint-Stereo-Phasenartefakte als Zeitversatz).

Dieser Test stellt sicher, dass:
  A) ALLE bekannten STCG-Aufrufpfade lag-freies Stereo UNVERÄNDERT durchlassen.
  B) Datei-Import keinen Lag auf sauberen MP3s einführt.
  C) Die simulierten Pipeline-Call-Chains keinen kumulativen Lag aufbauen.

Test-Daten: synthetisches Stereo mit exakt L==R (Simulation eines perfekt
korrelierten Stereosignals ohne Laufzeitdifferenz) sowie optional echte
Audio-Dateien aus corpus/.
"""

from __future__ import annotations

import numpy as np
import pytest

SR: int = 48_000

# ── Alle bekannten STCG phase_ids im Codebase ──────────────────────────
# Erfasst via grep correct_interchannel_delay backend/ am 2026-03
_ALL_STCG_PHASE_IDS: list[str] = [
    "import_pipeline",  # file_import.py:618
    "pre_pipeline",  # unified_restorer_v3.py:10924
    "phase_12_pre_chunking",  # phase_12_wow_flutter_fix.py:466
    "phase_12_wow_flutter_fix",  # phase_12_wow_flutter_fix.py:1517
    "phase_24",  # phase_24_dropout_repair.py:1415
    "phase_31",  # phase_31_speed_pitch_correction.py:799
    "post_pipeline",  # unified_restorer_v3.py:12473
    "stereo_drift_final",  # stereo_drift_state.py:114
    "post_export",  # unified_restorer_v3.py:13100
    "post_phase",  # unified_restorer_v3.py:9101
]


def _make_zero_lag_stereo(n_sec: float = 30.0, channels_first: bool = True) -> np.ndarray:
    """Erzeugt perfekt L==R Stereosignal ohne Laufzeitdifferenz."""
    rng = np.random.default_rng(42)
    n = int(SR * n_sec)
    mono = (rng.standard_normal(n) * 0.3).astype(np.float32)
    if channels_first:
        return np.vstack([mono[np.newaxis, :], mono[np.newaxis, :]])  # (2, N)
    else:
        return np.column_stack([mono, mono])  # (N, 2)


# ──────────────────────────────────────────────────────────────────────
# A) Alle STCG-Aufrufpfade — Null-Lag muss unverändert bleiben
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestAllSTCGPhaseIdsZeroLag:
    """Jede phase_id, die in der Codebase verwendet wird, muss lag-freies
    Stereo unverändert durchlassen."""

    def setup_method(self):
        from backend.core.stereo_temporal_coherence_guard import (
            get_stereo_temporal_coherence_guard,
        )

        self.stcg = get_stereo_temporal_coherence_guard()

    @pytest.mark.parametrize("phase_id", _ALL_STCG_PHASE_IDS)
    def test_zero_lag_passes_through(self, phase_id: str):
        audio_in = _make_zero_lag_stereo(30.0, channels_first=True)
        audio_out = self.stcg.correct_interchannel_delay(audio_in, SR, phase_id=phase_id)
        np.testing.assert_array_equal(
            audio_out,
            audio_in,
            err_msg=(
                f"STCG [{phase_id}] hat lag-freies Stereo verändert — False-Positive-Korrektur auf sauberem Input"
            ),
        )

    @pytest.mark.parametrize("phase_id", _ALL_STCG_PHASE_IDS)
    def test_zero_lag_channels_last_passes_through(self, phase_id: str):
        """Auch channels-last (N,2) Orientierung muss intakt bleiben."""
        audio_in = _make_zero_lag_stereo(30.0, channels_first=False)
        audio_out = self.stcg.correct_interchannel_delay(audio_in, SR, phase_id=phase_id)
        np.testing.assert_array_equal(
            audio_out,
            audio_in,
            err_msg=(f"STCG [{phase_id}] channels-last: lag-freies Stereo verändert"),
        )


# ──────────────────────────────────────────────────────────────────────
# B) Datei-Import: lag-freies Material darf keinen Lag bekommen
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestFileImportZeroLag:
    """Der Datei-Import (load_audio_file) darf auf sauberen Dateien keinen
    inter-channel Lag einführen oder fälschlich „korrigieren"."""

    def test_synthetic_wav_import_preserves_zero_lag(self, tmp_path):
        """Import einer synthetischen WAV-Datei (L==R) → Output L==R."""
        import scipy.io.wavfile as wavfile

        # 5 Sekunden Stereo-WAV schreiben
        filename = tmp_path / "zero_lag.wav"
        stereo = _make_zero_lag_stereo(5.0, channels_first=False)
        wavfile.write(str(filename), SR, (stereo * 0.9).astype(np.float32))

        from backend.file_import import load_audio_file

        result = load_audio_file(str(filename), target_sr=SR)
        audio_out = np.asarray(result["audio"], dtype=np.float32)  # type: ignore[index]

        # Kein NaN/Inf
        assert np.all(np.isfinite(audio_out)), "Import produziert NaN/Inf"

        # L und R müssen identisch sein (kein Lag eingeführt)
        if audio_out.ndim == 2:
            if audio_out.shape[0] == 2:  # channels-first
                np.testing.assert_allclose(
                    audio_out[0],
                    audio_out[1],
                    atol=1e-6,
                    err_msg="Import hat L/R-Differenz eingeführt (channels-first)",
                )
            elif audio_out.shape[1] == 2:  # channels-last
                np.testing.assert_allclose(
                    audio_out[:, 0],
                    audio_out[:, 1],
                    atol=1e-6,
                    err_msg="Import hat L/R-Differenz eingeführt (channels-last)",
                )


# ──────────────────────────────────────────────────────────────────────
# C) Simulierte Pipeline-Call-Chain — kein kumulativer Lag
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPipelineCallChainSimulation:
    """Die exakte Sequenz der STCG-Aufrufe einer realen Aurik-Pipeline
    darf auf lag-freiem Material keinen kumulativen Lag aufbauen."""

    def setup_method(self):
        from backend.core.stereo_temporal_coherence_guard import (
            get_stereo_temporal_coherence_guard,
        )

        self.stcg = get_stereo_temporal_coherence_guard()

    def test_full_pipeline_call_chain_preserves_zero_lag(self):
        """Simuliert: import → pre_pipeline → Phase 12 (pre+post) →
        post_pipeline (3× retry) → stereo_drift_final"""
        audio = _make_zero_lag_stereo(30.0, channels_first=True)
        original = audio.copy()

        # Step 1: Import-Pipeline STCG (file_import.py)
        audio = self.stcg.correct_interchannel_delay(audio, SR, phase_id="import_pipeline")

        # Step 2: Pre-Pipeline STCG (unified_restorer_v3.py)
        audio = self.stcg.correct_interchannel_delay(audio, SR, phase_id="pre_pipeline")

        # Step 3: Phase 12 Pre-Chunking (phase_12_wow_flutter_fix.py)
        audio = self.stcg.correct_interchannel_delay(audio, SR, phase_id="phase_12_pre_chunking")

        # Step 4: Phase 12 Post-WowFlutter (loudness preservation)
        audio = self.stcg.correct_interchannel_delay(audio, SR, phase_id="phase_12_wow_flutter_fix")

        # Step 5: Post-Pipeline mit 3 Retries (G14 loop)
        for _ in range(3):
            audio = self.stcg.correct_interchannel_delay(audio, SR, phase_id="post_pipeline")

        # Step 6: Phase 24 Dropout Repair
        audio = self.stcg.correct_interchannel_delay(audio, SR, phase_id="phase_24")

        # Step 7: Phase 31 Speed/Pitch Correction
        audio = self.stcg.correct_interchannel_delay(audio, SR, phase_id="phase_31")

        # Step 8: Stereo Drift Final
        audio = self.stcg.correct_interchannel_delay(audio, SR, phase_id="stereo_drift_final")

        # Step 9: Post-Export
        audio = self.stcg.correct_interchannel_delay(audio, SR, phase_id="post_export")

        # Finale Prüfung: Nach ALLEN Aufrufen muss das Signal unverändert sein
        np.testing.assert_array_equal(
            audio,
            original,
            err_msg=(
                "Pipeline-Call-Chain hat lag-freies Stereo korrumpiert — "
                "kumulative False-Positive-Korrekturen haben echten Lag ERZEUGT"
            ),
        )

    def test_post_pipeline_retry_loop_preserves_zero_lag(self):
        """G14 Retry-Loop (bis zu 3× post_pipeline) muss stabil sein."""
        audio = _make_zero_lag_stereo(30.0)
        original = audio.copy()

        from backend.file_import import _estimate_interchannel_lag_multi_point

        for retry in range(3):
            lag_profile = _estimate_interchannel_lag_multi_point(audio, SR, num_points=3)
            lag_pre = lag_profile["median_lag"]
            if abs(lag_pre) <= 50:
                break  # below threshold — done (simuliert G14)
            audio = self.stcg.correct_interchannel_delay(audio, SR, phase_id="post_pipeline")

        # Auf lag-freiem Material sollte die Loop sofort terminieren (lag_pre ~0)
        # und NICHTS korrigieren.
        np.testing.assert_array_equal(audio, original, err_msg="G14 Retry-Loop hat lag-freies Stereo verändert")
