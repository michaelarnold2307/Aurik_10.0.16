"""Test-Fixtures für Plugin-Entwicklung — VirtualAurikPipeline.

§15.6: Erlaubt isolierte Plugin-Tests ohne vollständige Aurik-Pipeline.

Usage::

    from plugins.sdk.testing_fixtures import VirtualAurikPipeline, make_test_audio

    pipeline = VirtualAurikPipeline(material="vinyl", era=1972)
    audio = make_test_audio(duration_s=3.0, sr=48000)
    result = pipeline.run_plugin(my_plugin, audio)

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def make_test_audio(
    duration_s: float = 3.0,
    sr: int = 48000,
    channels: int = 1,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Erzeugt synthetisches Test-Audio (Sinuston + Obertöne).

    Args:
        duration_s: Dauer in Sekunden.
        sr:         Abtastrate.
        channels:   Anzahl Kanäle (1 oder 2).
        frequency:  Grundfrequenz.
        amplitude:  Amplitude (0–1).

    Returns:
        float32 np.ndarray, shape=(samples,) oder (samples, channels).
    """
    samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, samples, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    signal += amplitude * 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
    signal += amplitude * 0.25 * np.sin(2 * np.pi * frequency * 3 * t)

    if channels == 2:
        signal = np.column_stack([signal, signal * 0.95])

    return signal.astype(np.float32)  # type: ignore[no-any-return]


def make_noisy_audio(
    duration_s: float = 3.0,
    sr: int = 48000,
    channels: int = 1,
    snr_db: float = 10.0,
    clicks_per_sec: float = 3.0,
) -> np.ndarray:
    """Erzeugt verrauschtes Test-Audio mit simulierten Defekten.

    Args:
        duration_s:     Dauer.
        sr:             Abtastrate.
        channels:       Kanäle.
        snr_db:         Signal-Rausch-Abstand.
        clicks_per_sec: Klicks/Sekunde (simulierte Vinyl-Defekte).

    Returns:
        float32 np.ndarray.
    """
    clean = make_test_audio(duration_s, sr, channels)
    signal_power = np.mean(clean**2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    rng = np.random.RandomState(42)
    if channels == 1:
        noise = np.sqrt(noise_power) * rng.randn(len(clean))
        result = clean + noise
    else:
        noise = np.sqrt(noise_power) * rng.randn(*clean.shape)
        result = clean + noise

    n_clicks = int(duration_s * clicks_per_sec)
    if channels == 1:
        for _ in range(n_clicks):
            pos = rng.randint(0, len(result) - 20)
            result[pos : pos + 5] += 0.5 * rng.randn(5)
    else:
        for _ in range(n_clicks):
            pos = rng.randint(0, result.shape[0] - 20)
            result[pos : pos + 5, :] += 0.5 * rng.randn(5, channels)

    return result.astype(np.float32)  # type: ignore[no-any-return]


@dataclass
class MockMaterialInfo:
    """Mock für Aurik-Material-Info."""

    material: str = "vinyl"
    era: int = 1972
    genre: str = "Jazz"
    defects: list[str] = field(default_factory=lambda: ["clicks", "surface_noise"])
    sample_rate: int = 48000
    channels: int = 2


@dataclass
class VirtualPipelineResult:
    """Ergebnis einer virtuellen Pipeline-Ausführung."""

    audio: np.ndarray
    sample_rate: int
    phase_results: list[dict] = field(default_factory=list)
    total_time_s: float = 0.0
    success: bool = True
    error: str = ""


class VirtualAurikPipeline:
    """Minimale Pipeline-Simulation für Plugin-Tests.

    Simuliert die Aurik-Pipeline-Umgebung ohne echte Phasen.
    Plugins können isoliert getestet werden.

    Usage::

        pipeline = VirtualAurikPipeline(material="vinyl")
        result = pipeline.run_plugin(my_plugin, test_audio)
        assert result.success
    """

    def __init__(
        self,
        material: str = "vinyl",
        era: int = 1972,
        genre: str = "Jazz",
        sample_rate: int = 48000,
    ):
        self.material_info = MockMaterialInfo(
            material=material,
            era=era,
            genre=genre,
            sample_rate=sample_rate,
        )

    def run_plugin(
        self,
        plugin,
        audio: np.ndarray,
        sr: int = 48000,
        **kwargs,
    ) -> VirtualPipelineResult:
        """Führt ein Plugin mit simulierter Pipeline-Umgebung aus.

        Args:
            plugin: AurikPlugin-Instanz.
            audio:  Eingabe-Audio.
            sr:     Abtastrate.
            **kwargs: Zusätzliche Parameter.

        Returns:
            VirtualPipelineResult mit Ergebnis.
        """
        import time

        t0 = time.perf_counter()

        try:
            ok, msg = plugin.validate()
            if not ok:
                return VirtualPipelineResult(
                    audio=audio,
                    sample_rate=sr,
                    success=False,
                    error=f"Plugin validation failed: {msg}",
                )

            plugin.on_phase_start(audio, sr, "virtual_phase", **kwargs)
            result_audio = plugin.process_audio(audio, sr, **kwargs)
            plugin.on_phase_end(result_audio, sr, "virtual_phase", **kwargs)

            elapsed = time.perf_counter() - t0
            return VirtualPipelineResult(
                audio=result_audio,
                sample_rate=sr,
                success=True,
                total_time_s=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return VirtualPipelineResult(
                audio=audio,
                sample_rate=sr,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                total_time_s=elapsed,
            )

    def get_material_info(self) -> MockMaterialInfo:
        """Gibt die Mock-Material-Info zurück."""
        return self.material_info


__all__ = [
    "make_test_audio",
    "make_noisy_audio",
    "MockMaterialInfo",
    "VirtualAurikPipeline",
    "VirtualPipelineResult",
]
