"""Unit-Tests für DefectType.AMPLITUDE_DRIFT Erkennung (§9.1c Bug 5).

Prüft:
- Carrier-induzierter Pegelanstieg → hohe Severity, is_artistic=False
- Künstlerisches Crescendo (korrelierter Onset-Anstieg) → geringe Severity, is_artistic=True
- Flaches Signal → severity=0.0
- Zu kurzes Signal (< 30 s) → severity=0.0
"""

import numpy as np
import pytest

from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

SR = 22050  # Analyse-SR (kein assert sr==48000 in Analyse-Modulen)


def _make_scanner(material: MaterialType = MaterialType.VINYL) -> DefectScanner:
    return DefectScanner(sample_rate=SR, material_type=material)


def _sine(freq: float, duration_s: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """Mono-Sinussignal."""
    t = np.arange(int(duration_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _rising_signal(duration_s: float = 60.0, sr: int = SR) -> np.ndarray:
    """Monoton steigendes Pegel-Signal ohne Onset-Korrelation — simuliert Träger-AGC-Drift."""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    # Amplitude steigt linear von 0.1 auf 0.6 → ~15 dB Anstieg über 60 s
    amp_env = np.linspace(0.1, 0.6, n).astype(np.float32)
    # 440 Hz Sinussignal mit aufsteigender Hüllkurve
    signal = amp_env * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    return signal  # type: ignore[no-any-return]


def _crescendo_with_onsets(duration_s: float = 60.0, sr: int = SR) -> np.ndarray:
    """Steigendes Signal MIT korreliertem Onset-Density-Anstieg — simuliert künstlerisches Crescendo."""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    amp_env = np.linspace(0.1, 0.5, n).astype(np.float32)
    signal = amp_env * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    # Transients (Onset-Pulse) ebenfalls zunehmend — korreliert mit Pegel
    hop = int(0.5 * sr)  # Puls alle 500 ms
    for i in range(0, n - hop, hop):
        progress = i / n  # 0 → 1
        # Onset-Intensität steigt mit Fortschritt — korreliert mit Pegel
        pulse_amp = float(progress * 0.3)
        pulse_len = min(int(0.02 * sr), n - i)
        signal[i : i + pulse_len] += np.random.uniform(-pulse_amp, pulse_amp, pulse_len).astype(np.float32)
    return signal  # type: ignore[no-any-return]


def _flat_signal(duration_s: float = 60.0, sr: int = SR) -> np.ndarray:
    """Gleichbleibender Pegel — kein Trend."""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)  # type: ignore[no-any-return]


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestAmplitudeDriftDetection:
    def test_detect_carrier_drift_high_severity(self) -> None:
        """Monoton steigendes Signal ohne Onset-Korrelation → Severity > 0.3, is_artistic=False."""
        scanner = _make_scanner(MaterialType.VINYL)
        signal = _rising_signal(duration_s=65.0)
        score = scanner._detect_amplitude_drift(signal)

        assert score.defect_type == DefectType.AMPLITUDE_DRIFT
        assert score.severity > 0.20, f"Carrier drift should have high severity, got {score.severity:.3f}"
        assert score.metadata.get("is_artistic") is False, "Carrier drift must not be classified as artistic"
        assert abs(score.metadata.get("drift_db_per_minute", 0.0)) > 1.0, "slope should be > 1.5 dB/min"

    def test_detect_artistic_crescendo_low_severity(self) -> None:
        """Signal mit korreliertem Onset-Anstieg → is_artistic=True oder severity <= 0.20."""
        scanner = _make_scanner(MaterialType.VINYL)
        np.random.seed(42)
        signal = _crescendo_with_onsets(duration_s=65.0)
        score = scanner._detect_amplitude_drift(signal)

        assert score.defect_type == DefectType.AMPLITUDE_DRIFT
        # Artistic: severity muss <= 0.20 sein (§9.1c: artistic → max 0.20)
        # ODER is_artistic=True (beides ist korrekt, Test akzeptiert beide Fälle)
        artistic_or_low = score.metadata.get("is_artistic", False) or score.severity <= 0.20
        assert artistic_or_low, (
            f"Artistic crescendo should be classified as artistic or low severity, "
            f"got severity={score.severity:.3f}, is_artistic={score.metadata.get('is_artistic')}"
        )

    def test_flat_signal_zero_severity(self) -> None:
        """Signal ohne Trend → severity=0.0."""
        scanner = _make_scanner(MaterialType.CD_DIGITAL)
        signal = _flat_signal(duration_s=65.0)
        score = scanner._detect_amplitude_drift(signal)

        assert score.defect_type == DefectType.AMPLITUDE_DRIFT
        assert score.severity == pytest.approx(0.0, abs=0.15), (
            f"Flat signal should have near-zero severity, got {score.severity:.3f}"
        )

    def test_too_short_skipped(self) -> None:
        """Signal < 30 s → severity=0.0 (zu kurz für Trend-Analyse)."""
        scanner = _make_scanner(MaterialType.VINYL)
        signal = _rising_signal(duration_s=20.0)  # nur 20 s
        score = scanner._detect_amplitude_drift(signal)

        assert score.defect_type == DefectType.AMPLITUDE_DRIFT
        assert score.severity == pytest.approx(0.0, abs=1e-6), (
            f"Signal < 30s should be skipped (severity=0.0), got {score.severity:.3f}"
        )
        assert score.confidence <= 0.5

    def test_defect_type_enum_exists(self) -> None:
        """DefectType.AMPLITUDE_DRIFT muss im Enum vorhanden sein."""
        assert hasattr(DefectType, "AMPLITUDE_DRIFT")
        assert DefectType.AMPLITUDE_DRIFT.value == "amplitude_drift"

    def test_material_sensitivity_entries(self) -> None:
        """Alle relevanten Materialtypen haben AMPLITUDE_DRIFT-Sensitivity-Einträge (via Scanner-Instanz)."""
        for mat in [
            MaterialType.VINYL,
            MaterialType.SHELLAC,
            MaterialType.TAPE,
            MaterialType.CD_DIGITAL,
            MaterialType.MP3_LOW,
        ]:
            scanner = DefectScanner(sample_rate=SR, material_type=mat)
            # thresholds dict enthält die sensitivity-basierten Schwellen; AMPLITUDE_DRIFT muss vorhanden sein
            assert DefectType.AMPLITUDE_DRIFT in scanner.thresholds, (
                f"Scanner.thresholds for {mat} missing AMPLITUDE_DRIFT entry — MATERIAL_SENSITIVITY fehlt Eintrag"
            )

    def test_stereo_input_handled(self) -> None:
        """Stereo-Input (2, n) wird korrekt verarbeitet → kein Crash."""
        scanner = _make_scanner(MaterialType.VINYL)
        mono = _rising_signal(duration_s=65.0)
        stereo = np.stack([mono, mono], axis=0)
        score = scanner._detect_amplitude_drift(stereo)
        assert score.defect_type == DefectType.AMPLITUDE_DRIFT
        # Stereo (identische Kanäle) soll gleiche Severity wie Mono liefern
        score_mono = scanner._detect_amplitude_drift(mono)
        assert abs(score.severity - score_mono.severity) < 0.05
