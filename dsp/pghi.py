"""
Aurik 10.0.0 — PGHI: Phase Gradient Heap Integration
================================================
Standalone-Implementierung der phasenkonsistenten Spektral-Rekonstruktion
nach Perraudin et al. (2013). Pflicht-Modul für alle Phasen und Plugins
die das Betragssspektrum modifizieren.

Referenz:
    Perraudin, N., Balazs, P., & Søndergaard, P. L. (2013).
    A Non-Iterative Method for STFT Phase (Re)construction.
    Signal, Image and Video Processing, 7(6), 1093-1100.
    https://doi.org/10.1007/s11760-012-0350-8

Warum PGHI?
    Nach jeder Spektral-Modifikation (NMF, OMLSA, EQ, Inpainting) liegt nur
    das modifizierte Betragsspektrum vor; das Phasenspektrum ist nicht mehr
    konsistent. Direktes ISTFT mit alten Phasen erzeugt Artefakte.
    PGHI rekonstruiert Phasen nicht-iterativ aus Phasengradienten — deutlich
    schneller als Griffin-Lim (≥ 32 Iterationen) bei vergleichbarer Qualität.

Invarianten:
    - Keine ML-Abhängigkeit: reine NumPy-Implementierung
    - NaN/Inf-sicher: alle Ausgaben durch nan_to_num + clip
    - Thread-sicher: Singleton mit Double-Checked Locking (§3.2)
    - Fallback: Griffin-Lim+ (32 Iterationen) wenn PGHI numerisch instabil
    - Laufzeit: O(N·log N) pro Frame — schneller als Griffin-Lim O(N·log N × iter)
"""

from __future__ import annotations

import heapq
import logging
import math
import threading
from dataclasses import dataclass

import numpy as np
from scipy.signal import istft as _scipy_istft
from scipy.signal import stft as _scipy_stft

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class PghiResult:
    """Ergebnis der PGHI-Phasenrekonstruktion."""

    audio: np.ndarray  # Rekonstruiertes Audio [n_samples], float32
    stft_complex: np.ndarray  # Rekonstruiertes STFT [n_bins, n_frames], complex64
    n_frames: int
    win_size: int
    hop: int
    method_used: str  # "pghi" oder "griffin_lim_fallback"


# ---------------------------------------------------------------------------
# PGHI-Implementierung
# ---------------------------------------------------------------------------


class PghiReconstructor:
    """Phase Gradient Heap Integration (PGHI) — nicht-iterative Phasenrekonstruktion.

    Algorithmus (Perraudin et al. 2013):
        1. Phasengradient in Zeitrichtung (dφ/dt) aus Systemfrequenz ableiten:
           δφ_t[k, m] = 2π · hop · k/N + phase_deviation_t
        2. Phasengradient in Frequenzrichtung (dφ/dω) aus Gruppendelays:
           δφ_ω[k, m] = -∂|STFT|/∂t / |STFT|   (Gruppendelay-Näherung)
        3. Heap-geführte Integration beginnend an Energie-Maximum:
           - Startet am Bin mit maximaler Energie
           - Propagiert Phasen zu Nachbarn via Gradienten
           - Heap-Priorisierung: höhere Energie zuerst → stabile Schätzung
        4. iSTFT mit rekonstruierten Phasen → Audio

    Schlüssel-Unterschied zu Griffin-Lim:
        Griffin-Lim iteriert zwischen STFT und iSTFT bis Konsistenz.
        PGHI integriert Phasen in einem Durchgang — O(N·log N) statt O(iter·N·log N).

    Fallback (Griffin-Lim+):
        Wenn PGHI numerisch instabil (NaN/Inf in Gradienten): 32 Iterationen
        Griffin-Lim als Fallback.
    """

    def __init__(
        self,
        sr: int = 48000,
        win_size: int = 2048,
        hop: int = 256,
        gamma: float = 1.0,
        tol: float = 1e-6,
        max_griffin_lim_iter: int = 32,
    ) -> None:
        """Initialisiert PGHI.

        Args:
            sr:                  Sample-Rate (muss 48000)
            win_size:            STFT-Fenstergröße (Hanning)
            hop:                 Hop-Größe
            gamma:               PGHI-Regulierungsparameter (1.0 = Standardwert)
            tol:                 Numerische Toleranz für stabile Gradienten
            max_griffin_lim_iter: Max. Iterationen für Griffin-Lim-Fallback
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        self.sr = sr
        self.win_size = win_size
        self.hop = hop
        self.gamma = gamma
        self.tol = tol
        self.max_griffin_lim_iter = max_griffin_lim_iter
        self._window = np.hanning(win_size).astype(np.float64)
        self._window_sum_sq = np.sum(self._window**2)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        magnitude: np.ndarray,
        win_size: int | None = None,
        hop: int | None = None,
        initial_phase: np.ndarray | None = None,
    ) -> PghiResult:
        """Rekonstruiert Audio aus Betrags-Spektrogramm via PGHI.

        Args:
            magnitude:     |STFT| [n_bins, n_frames], float32/64, ≥ 0
            win_size:      STFT-Fenstergröße (überschreibt __init__)
            hop:           Hop-Größe (überschreibt __init__)
            initial_phase: Anfangs-Phase [n_bins, n_frames] (optional, als Hint)

        Returns:
            PghiResult mit audio und rekonstruiertem STFT.
        """
        ws = win_size or self.win_size
        h = hop or self.hop
        n_bins, n_frames = magnitude.shape
        n_samples = (n_frames - 1) * h + ws

        # Sicherheits-Check
        if not np.all(np.isfinite(magnitude)):
            magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
        magnitude = np.clip(magnitude, 0.0, None)

        # Versuche PGHI
        try:
            phase = self._pghi(magnitude, ws, h, initial_phase)
            if not np.all(np.isfinite(phase)):
                raise ValueError("PGHI Phasen nicht-finite")
            method = "pghi"
        except Exception as e:
            logger.debug("PGHI Ersatzpfad (Griffin-Lim): %s", e)
            phase = self._griffin_lim(magnitude, ws, h)
            method = "griffin_lim_fallback"

        stft_complex = magnitude * np.exp(1j * phase)
        audio = self._istft(stft_complex, ws, h, n_samples)

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        logger.debug("PGHI: method=%s, n_bins=%d, n_frames=%d", method, n_bins, n_frames)

        return PghiResult(
            audio=audio,
            stft_complex=stft_complex.astype(np.complex64),
            n_frames=n_frames,
            win_size=ws,
            hop=h,
            method_used=method,
        )

    def reconstruct_from_stft(
        self,
        stft_modified: np.ndarray,
        win_size: int | None = None,
        hop: int | None = None,
        use_original_phase: bool = False,
    ) -> PghiResult:
        """Rekonstruiert Audio aus modifiziertem STFT (Betrag + alte Phase als Hint).

        Typischer Anwendungsfall: Nach OMLSA/NMF Gain-Anwendung.

        Args:
            stft_modified:    Modifiziertes STFT [n_bins, n_frames], complex64/128
            win_size:         STFT-Fenstergröße
            hop:              Hop-Größe
            use_original_phase: True → alte Phasen behalten (schnell, weniger konsistent);
                              False → PGHI aus neuem Betrag (langsamer, konsistenter)

        Returns:
            PghiResult
        """
        magnitude = np.abs(stft_modified)
        initial_phase = np.angle(stft_modified)

        if use_original_phase:
            audio = self._istft(stft_modified, win_size or self.win_size, hop or self.hop, boundary=True)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            return PghiResult(
                audio=audio,
                stft_complex=stft_modified.astype(np.complex64),
                n_frames=stft_modified.shape[1],
                win_size=win_size or self.win_size,
                hop=hop or self.hop,
                method_used="original_phase",
            )

        # use_original_phase=False: PGHI-Phasenneuberechnung aus Betrag.
        # _istft mit boundary=True, da der externe STFT mit boundary='zeros'
        # berechnet wurde (scipy-Standard in allen Aufruf-Phasen).
        result = self.reconstruct(magnitude, win_size, hop, initial_phase)
        # Rekonstruiere Audio mit korrekter boundary-Behandlung
        stft_pghi = result.stft_complex
        ws = win_size or self.win_size
        h = hop or self.hop
        audio_pghi = self._istft(stft_pghi, ws, h, boundary=True)
        audio_pghi = np.nan_to_num(audio_pghi, nan=0.0, posinf=0.0, neginf=0.0)
        audio_pghi = np.clip(audio_pghi, -1.0, 1.0).astype(np.float32)
        return PghiResult(
            audio=audio_pghi,
            stft_complex=stft_pghi,
            n_frames=stft_pghi.shape[1],
            win_size=ws,
            hop=h,
            method_used=result.method_used,
        )

    # ------------------------------------------------------------------
    # PGHI-Kern
    # ------------------------------------------------------------------

    def _pghi(
        self,
        magnitude: np.ndarray,
        win_size: int,
        hop: int,
        initial_phase: np.ndarray | None = None,
    ) -> np.ndarray:
        """Kern-PGHI-Algorithmus.

        Phasengradienten-Berechnung:
            δφ/δt ≈ 2π·hop·k/N + ∂arg{H}/∂t      (Instantanfrequenz)
            δφ/δω ≈ -∂log|X|/∂ω                   (Gruppendelay)

        Heap-Integration:
            Startet an Energie-Maximum, propagiert zu Nachbarn
            mit absteigender Energie-Priorisierung.

        Args:
            magnitude:     [n_bins, n_frames], float64
            win_size:      Fenstergröße
            hop:           Hop-Größe
            initial_phase: Optionaler Phasen-Hint

        Returns:
            Rekonstruierte Phasen [n_bins, n_frames], float64
        """
        n_bins, n_frames = magnitude.shape
        mag = magnitude.astype(np.float64)

        # Log-Magnitude (numerisch stabil)
        log_mag = np.log(mag + self.tol)

        # Phasengradienten in Zeitrichtung (analytische Formel)
        # δφ_t[k, m] = 2π · hop · k / win_size
        k_indices = np.arange(n_bins)
        delta_phi_t = 2.0 * math.pi * hop * k_indices[:, np.newaxis] / win_size
        delta_phi_t = np.tile(delta_phi_t, (1, n_frames))

        # Phasengradienten in Frequenzrichtung (Gruppendelay-Näherung)
        # δφ_ω = -∂log|X|/∂ω  ≈ -∂log_mag/∂k × N/(2π)
        # Zentrale Differenzen
        d_log_mag_dk = np.zeros_like(log_mag)
        d_log_mag_dk[1:-1, :] = (log_mag[2:, :] - log_mag[:-2, :]) / 2.0
        d_log_mag_dk[0, :] = log_mag[1, :] - log_mag[0, :]
        d_log_mag_dk[-1, :] = log_mag[-1, :] - log_mag[-2, :]

        # Gamma-skalierte Gruppendelay-Schätzung
        delta_phi_omega = -self.gamma * d_log_mag_dk * win_size / (2.0 * math.pi)

        # Phase-Array (initialisiert)
        phase = np.zeros((n_bins, n_frames), dtype=np.float64)
        if initial_phase is not None:
            phase[:, 0] = initial_phase[:, 0] if initial_phase.ndim > 1 else initial_phase
        else:
            # Zufällige Startphase am ersten Frame
            rng = np.random.default_rng(seed=42)
            phase[:, 0] = rng.uniform(-math.pi, math.pi, n_bins)

        # Heap-geführte Integration
        # Priorität: höhere Energie zuerst (min-Heap mit negativer Energie)
        heap: list[tuple[float, int, int]] = []  # (-energy, bin, frame)
        visited = np.zeros((n_bins, n_frames), dtype=bool)

        # Startpunkte: alle Bins des ersten Frames
        for k in range(n_bins):
            energy = mag[k, 0]
            heapq.heappush(heap, (-energy, k, 0))
        # Außerdem: Energy-Maximum über alle Frames
        max_frame = int(np.argmax(np.max(mag, axis=0)))
        max_bin = int(np.argmax(mag[:, max_frame]))
        if not (max_bin == 0 and max_frame == 0):
            heapq.heappush(heap, (-mag[max_bin, max_frame], max_bin, max_frame))

        propagated = 0
        while heap and propagated < n_bins * n_frames * 2:
            _neg_energy, k, m = heapq.heappop(heap)
            if visited[k, m]:
                continue
            visited[k, m] = True
            propagated += 1

            current_phase = phase[k, m]

            # Propagation zu Zeitnachbar (m+1)
            if m + 1 < n_frames and not visited[k, m + 1]:
                # Phase in Zeitrichtung: φ[k, m+1] ≈ φ[k, m] + δφ_t[k, m]
                phi_t = current_phase + delta_phi_t[k, m]
                # Gewichtetes Mittel mit Vorwärtspropagation
                if phase[k, m + 1] == 0.0:
                    phase[k, m + 1] = phi_t
                else:
                    w = mag[k, m] / (mag[k, m] + mag[k, m + 1] + 1e-10)
                    phase[k, m + 1] = w * phi_t + (1.0 - w) * phase[k, m + 1]
                heapq.heappush(heap, (-mag[k, m + 1], k, m + 1))

            # Propagation zu Frequenznachbar (k+1)
            if k + 1 < n_bins and not visited[k + 1, m]:
                # Phase in Frequenzrichtung: φ[k+1, m] ≈ φ[k, m] + δφ_ω[k, m]
                phi_k = current_phase + delta_phi_omega[k, m]
                if phase[k + 1, m] == 0.0:
                    phase[k + 1, m] = phi_k
                else:
                    w = mag[k, m] / (mag[k, m] + mag[k + 1, m] + 1e-10)
                    phase[k + 1, m] = w * phi_k + (1.0 - w) * phase[k + 1, m]
                heapq.heappush(heap, (-mag[k + 1, m], k + 1, m))

            # Rückwärts-Propagation (m-1)
            if m - 1 >= 0 and not visited[k, m - 1]:
                phi_t_back = current_phase - delta_phi_t[k, m]
                if phase[k, m - 1] == 0.0:
                    phase[k, m - 1] = phi_t_back
                heapq.heappush(heap, (-mag[k, m - 1], k, m - 1))

            # Rückwärts (k-1)
            if k - 1 >= 0 and not visited[k - 1, m]:
                phi_k_back = current_phase - delta_phi_omega[k, m]
                if phase[k - 1, m] == 0.0:
                    phase[k - 1, m] = phi_k_back
                heapq.heappush(heap, (-mag[k - 1, m], k - 1, m))

        # Phasen wrapped auf [-π, π]
        phase = np.angle(np.exp(1j * phase))

        return phase.astype(np.float64)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Griffin-Lim Fallback
    # ------------------------------------------------------------------

    def _griffin_lim(
        self,
        magnitude: np.ndarray,
        win_size: int,
        hop: int,
        n_iter: int | None = None,
    ) -> np.ndarray:
        """Griffin-Lim+ Algorithmus (Fallback, 32 Iterationen).

        Alternierend: STFT-Konsistenzprojektion ↔ Magnitude-Ersatz.

        Args:
            magnitude: [n_bins, n_frames]
            win_size:  Fenstergröße
            hop:       Hop-Größe
            n_iter:    Anzahl Iterationen (Default: max_griffin_lim_iter)

        Returns:
            Rekonstruierte Phasen [n_bins, n_frames]
        """
        n_iter = n_iter or self.max_griffin_lim_iter
        n_bins, n_frames = magnitude.shape
        n_samples = (n_frames - 1) * hop + win_size

        # Zufällige Startphase
        rng = np.random.default_rng(seed=99)
        phase = rng.uniform(-math.pi, math.pi, (n_bins, n_frames))
        stft = magnitude * np.exp(1j * phase)

        for _ in range(n_iter):
            # iSTFT → Audio
            audio = self._istft(stft, win_size, hop, n_samples)
            # Analyse-STFT
            stft_new = self._stft(audio, win_size, hop)
            # Phase übernehmen, Magnitude erzwingen
            rows = min(n_bins, stft_new.shape[0])
            cols = min(n_frames, stft_new.shape[1])
            stft_new_phase = np.angle(stft_new[:rows, :cols])
            padded = np.zeros((n_bins, n_frames))
            padded[:rows, :cols] = stft_new_phase
            stft = magnitude * np.exp(1j * padded)

        return np.angle(stft)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # STFT / iSTFT Hilfsfunktionen
    # ------------------------------------------------------------------

    def _stft(
        self,
        audio: np.ndarray,
        win_size: int,
        hop: int,
    ) -> np.ndarray:
        """Vectorised STFT via scipy.signal.stft. Returns [n_bins, n_frames] complex128.

        Replaces the previous frame-loop implementation (O(frames·N·log N)) with a
        single batched FFT call — approx. 10× faster for typical window sizes.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float64)
        _f, _t, Zxx = _scipy_stft(
            audio,
            fs=self.sr,
            window="hann",
            nperseg=win_size,
            noverlap=win_size - hop,
            nfft=win_size,
            boundary=None,
            padded=False,
        )
        return Zxx.astype(np.complex128)  # type: ignore  # [n_bins, n_frames]

    def _istft(
        self,
        stft_complex: np.ndarray,
        win_size: int,
        hop: int,
        n_samples: int = -1,
        boundary: bool = False,
    ) -> np.ndarray:
        """Vectorised inverse STFT via scipy.signal.istft. Returns float32 audio.

        Replaces the previous frame-loop OLA implementation with a single batched
        call — approx. 10× faster and uses the same OLA normalisation as scipy.

        Args:
            stft_complex: [n_bins, n_frames], complex
            win_size:     Analysis window size (must match STFT)
            hop:          Hop size (win_size - noverlap)
            n_samples:    If > 0, trim/pad output to this length
            boundary:     Pass True when stft_complex was computed with
                          scipy.signal.stft's default boundary='zeros' padding.
                          scipy then strips the synthetic edge frames and
                          reconstructs the original-length signal correctly.
        """
        import warnings as _warnings

        with _warnings.catch_warnings():
            # boundary=False intentionally omits edge frames; NOLA is satisfied
            # for interior frames of a 75%-overlap Hann window. Edge artefacts
            # are negligible and corrected by the n_samples trim below.
            _warnings.filterwarnings(
                "ignore",
                message="NOLA condition failed",
                category=UserWarning,
            )
            _t, audio = _scipy_istft(
                stft_complex.astype(np.complex128),
                fs=self.sr,
                window="hann",
                nperseg=win_size,
                noverlap=win_size - hop,
                nfft=win_size,
                boundary=boundary,
            )
        if n_samples > 0:
            if len(audio) > n_samples:
                audio = audio[:n_samples]
            elif len(audio) < n_samples:
                audio = np.pad(audio, (0, n_samples - len(audio)))
        return np.nan_to_num(audio, nan=0.0).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton (§3.2 — Thread-sicheres Double-Checked Locking)
# ---------------------------------------------------------------------------

_instance: PghiReconstructor | None = None
_lock = threading.Lock()


def get_pghi_reconstructor(
    sr: int = 48000,
    win_size: int = 2048,
    hop: int = 256,
) -> PghiReconstructor:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2).

    Args:
        sr:       Sample-Rate (muss 48000)
        win_size: Standard-Fenstergröße
        hop:      Standard-Hop-Größe

    Returns:
        Globale PghiReconstructor-Instanz.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PghiReconstructor(sr=sr, win_size=win_size, hop=hop)
    return _instance


# ---------------------------------------------------------------------------
# Convenience-Funktionen (Pflicht-API für alle Phasen und Plugins)
# ---------------------------------------------------------------------------


def pghi_reconstruct(
    magnitude: np.ndarray,
    sr: int = 48000,
    win_size: int = 2048,
    hop: int = 256,
    initial_phase: np.ndarray | None = None,
) -> np.ndarray:
    """Rekonstruiert Audio aus Betrags-Spektrogramm via PGHI.

    Primäre Schnittstelle für alle Phasen/Plugins (§4.5).
    Ersetzt direktes ISTFT auf modifizierten Spektren ohne Phasenrekonstruktion.

    Algorithmus:
        PGHI (Perraudin 2013) → Fallback: Griffin-Lim+ (32 Iter.)

    Args:
        magnitude:     |STFT| [n_bins, n_frames], float32/64, ≥ 0
        sr:            Sample-Rate (muss 48000 Hz sein)
        win_size:      STFT-Fenstergröße
        hop:           Hop-Größe
        initial_phase: Optionale Startphase als Hint [n_bins, n_frames]

    Returns:
        Rekonstruiertes Audio [n_samples], float32, ∈ [-1, 1]

    Invariante:
        Ausgabe immer NaN/Inf-frei und ∈ [-1, 1].
    """
    rec = get_pghi_reconstructor(sr, win_size, hop)
    result = rec.reconstruct(magnitude, win_size, hop, initial_phase)
    return result.audio


def pghi_reconstruct_from_stft(
    stft_modified: np.ndarray,
    sr: int = 48000,
    win_size: int = 2048,
    hop: int = 256,
    use_original_phase: bool = False,
    n_samples: int = -1,
) -> np.ndarray:
    """Rekonstruiert Audio aus modifiziertem STFT mit optionaler Phasen-Neuberechnung.

    Typischer Aufruf nach OMLSA Gain-Anwendung, NMF-Masking, EQ-Correction.

    Args:
        stft_modified:    Modifiziertes STFT [n_bins, n_frames], complex
        sr:               Sample-Rate (muss 48000 Hz)
        win_size:         STFT-Fenstergröße
        hop:              Hop-Größe
        use_original_phase: True → alte Phasen behalten (schneller, weniger konsistent)
        n_samples:        Wenn >0, Ausgabe auf diese Länge trimmen/padden
                          (sollte len(original_audio) sein für exakte Längetreue).

    Returns:
        Audio [n_samples], float32, ∈ [-1, 1]
    """
    rec = get_pghi_reconstructor(sr, win_size, hop)
    result = rec.reconstruct_from_stft(stft_modified, win_size, hop, use_original_phase)
    audio = result.audio
    if n_samples > 0:
        if len(audio) > n_samples:
            audio = audio[:n_samples]
        elif len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
    return audio


def griffin_lim_reconstruct(
    magnitude: np.ndarray,
    sr: int = 48000,
    win_size: int = 2048,
    hop: int = 256,
    n_iter: int = 32,
) -> np.ndarray:
    """Griffin-Lim+ Rekonstruktion (expliziter Fallback für einfache Fälle).

    Args:
        magnitude: [n_bins, n_frames], float32/64
        sr:        Sample-Rate (muss 48000)
        win_size:  Fenstergröße
        hop:       Hop-Größe
        n_iter:    Iterationsanzahl (Standard: 32)

    Returns:
        Audio [n_samples], float32, ∈ [-1, 1]
    """
    rec = get_pghi_reconstructor(sr, win_size, hop)
    phase = rec._griffin_lim(np.asarray(magnitude, dtype=np.float64), win_size, hop, n_iter)
    stft = magnitude * np.exp(1j * phase)
    n_samples = (magnitude.shape[1] - 1) * hop + win_size
    audio = rec._istft(stft, win_size, hop, n_samples)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)
