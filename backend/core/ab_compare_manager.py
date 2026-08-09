"""
core/ab_compare_manager.py — A/B-Vergleichs-Session-Manager (Aurik 10.0.0)

Speichert Original- und restauriertes Audio paarweise in einer Session-Datenbank,
damit der Nutzer jederzeit Vorher/Nachher vergleichen kann.

Fähigkeiten:
  • Speichert (original_audio, restored_audio) unter einer eindeutigen Session-ID
  • Berechnet sofort hörbare Verbesserungs-Kennzahlen (RMS-Delta, spektrale Differenz)
  • Hält maximal MAX_SESSIONS Sessions im RAM (LRU-artig)
  • Schreibt optional eine leichte JSON-Sidecar-Datei in ~/.aurik/ab_sessions/
  • Thread-sicherer Singleton (§3.2 Spec: Double-Checked Locking)

Keine GUI-Abhängigkeiten — rein backends-seitiger Dienst.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

_AB_SESSION_DIR = Path.home() / ".aurik" / "ab_sessions"
MAX_SESSIONS = 32  # LRU-Cache-Limit im RAM


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class ABDiff:
    """
    Quantitative Differenz zwischen Original und restauriertem Audio.

    Alle Werte sind laienverständlich interpretierbar:
        rms_improvement_db > 0  → restauriertes Audio hat bessere Dynamik
        snr_estimate_db         → Signal-Rausch-Verhältnis-Schätzung
        spectral_similarity     → 1.0 = klanglich identisch, 0.0 = völlig anders
    """

    rms_original_db: float
    rms_restored_db: float
    rms_improvement_db: float  # positiv = restauriert hat weniger Rauschen
    spectral_similarity: float  # ∈ [0, 1]  — NSIM-ähnlich auf Log-Spektrum
    snr_estimate_db: float  # Signal-Rausch-Verhältnis-Schätzung
    peak_original: float
    peak_restored: float
    duration_seconds: float
    n_channels: int

    def as_dict(self) -> dict:
        return {
            "rms_original_db": round(self.rms_original_db, 2),
            "rms_restored_db": round(self.rms_restored_db, 2),
            "rms_improvement_db": round(self.rms_improvement_db, 2),
            "spectral_similarity": round(self.spectral_similarity, 4),
            "snr_estimate_db": round(self.snr_estimate_db, 2),
            "peak_original": round(self.peak_original, 4),
            "peak_restored": round(self.peak_restored, 4),
            "duration_seconds": round(self.duration_seconds, 3),
            "n_channels": self.n_channels,
        }

    def human_verdict(self) -> str:
        """Laienverständliches Fazit auf Basis der Differenz-Kennzahlen."""
        if self.rms_improvement_db > 12:
            return "Dramatische Verbesserung — der Klangunterschied ist enorm."
        if self.rms_improvement_db > 6:
            return "Sehr deutliche Verbesserung — sofort hörbar."
        if self.rms_improvement_db > 2:
            return "Klare Verbesserung — im direkten Vergleich gut hörbar."
        if self.rms_improvement_db > 0:
            return "Spürbare Verbesserung — besonders bei Lärm und Rauschen."
        if self.rms_improvement_db > -1:
            return "Kein messbarer Unterschied im Lautstärkepegel (Qualität kann trotzdem besser sein)."
        return "Das Original und das restaurierte Audio sind klanglich sehr ähnlich."


@dataclass
class ABSession:
    """
    Ein gespeichertes Original/Restauriert-Paar für den A/B-Vergleich.

    Felder:
        session_id:     Eindeutige UUID für diese Session
        material:       Materialtyp (z.B. "vinyl")
        created_at:     Unix-Timestamp der Erstellung
        sample_rate:    Sample-Rate beider Audios
        diff:           Berechente Differenz-Metriken
        original_sha256: SHA256-Fingerabdruck des Originals (16 Hex)
        tags:           Optionale Nutzer-Tags
    """

    session_id: str
    material: str
    created_at: float
    sample_rate: int
    diff: ABDiff
    original_sha256: str
    tags: list[str] = field(default_factory=list)

    # Audio-Arrays — nicht serialisiert (nur im RAM)
    _original: np.ndarray | None = field(default=None, repr=False)
    _restored: np.ndarray | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "material": self.material,
            "created_at": round(self.created_at, 3),
            "sample_rate": self.sample_rate,
            "diff": self.diff.as_dict(),
            "original_sha256": self.original_sha256,
            "tags": self.tags,
            "human_verdict": self.diff.human_verdict(),
        }


# ---------------------------------------------------------------------------
# Haupt-Klasse: ABCompareManager
# ---------------------------------------------------------------------------


class ABCompareManager:
    """
    Verwaltet A/B-Vergleichs-Sessions für Original- und restauriertes Audio.

    Algorithmus:
        1. store(original, restored, sr, material) → session_id
           - SHA256-Fingerabdruck des Originals berechnen
           - ABDiff berechnen (RMS, spektrale Ähnlichkeit, SNR)
           - ABSession im RAM-Cache speichern (LRU, max MAX_SESSIONS)
           - JSON-Sidecar in ~/.aurik/ab_sessions/ schreiben (kein Audio!)
        2. get(session_id) → Optional[ABSession]
        3. list_sessions() → List[dict]
        4. compare_audio(session_id) → Optional[Tuple[np.ndarray, np.ndarray]]

    Thread-Sicherheit: Thread-sicher durch _lock (RLock für verschachtelte Aufrufe).
    """

    def __init__(self) -> None:
        # LRU-Cache: OrderedDict → session_id → ABSession
        self._sessions: OrderedDict[str, ABSession] = OrderedDict()
        self._lock = threading.RLock()
        _AB_SESSION_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("ABCompareManager initialisiert (max %d Sessions)", MAX_SESSIONS)

    # -------------------------------------------------------------------
    # Öffentliche API
    # -------------------------------------------------------------------

    def store(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sample_rate: int,
        material: str = "unknown",
        tags: list[str] | None = None,
    ) -> str:
        """
        Speichert ein Original/Restauriert-Paar und gibt die Session-ID zurück.

        Args:
            original:    Original-Audio (float32, 1D oder 2D [samples × channels])
            restored:    Restauriertes Audio (gleiche Form)
            sample_rate: Sample-Rate beider Audios in Hz
            material:    Materialtyp-String (z.B. "vinyl")
            tags:        Optionale Nutzer-Tags

        Returns:
            session_id (UUID-String) — zum späteren Abrufen

        Invarianten:
            - Kein NaN/Inf in Ein- oder Ausgabe toleriert (wird zu 0 ersetzt)
            - Kurzes Audio (< 0.1 s) wird akzeptiert, Diff bleibt sinnlos
        """
        original = np.nan_to_num(np.asarray(original, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        restored = np.nan_to_num(np.asarray(restored, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        session_id = str(uuid.uuid4())
        sha256_orig = self._sha256_audio(original)
        diff = self._compute_diff(original, restored, sample_rate)
        session = ABSession(
            session_id=session_id,
            material=material,
            created_at=time.time(),
            sample_rate=sample_rate,
            diff=diff,
            original_sha256=sha256_orig,
            tags=tags or [],
            _original=original.copy(),
            _restored=restored.copy(),
        )

        with self._lock:
            self._sessions[session_id] = session
            # LRU: älteste Session entfernen wenn Limit erreicht
            while len(self._sessions) > MAX_SESSIONS:
                removed_id, _ = self._sessions.popitem(last=False)
                logger.debug("ABCompare: LRU-Entfernung von Sitzung %s", removed_id[:8])

        # JSON-Sidecar schreiben (kein Audio, nur Metriken)
        self._write_sidecar(session)

        logger.info(
            "📊 A/B-Sitzung %s: %s | RMS-Delta %.1f dB | Ähnlichkeit %.2f",
            session_id[:8],
            material,
            diff.rms_improvement_db,
            diff.spectral_similarity,
        )
        return session_id

    def get(self, session_id: str) -> ABSession | None:
        """Gibt eine gespeicherte A/B-Session zurück, oder None wenn nicht gefunden."""
        with self._lock:
            return self._sessions.get(session_id)

    def compare_audio(self, session_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Gibt (original_audio, restored_audio) zurück — für Frontend-Wiedergabe.

        Returns:
            Tuple (original, restored) oder None wenn Session nicht gefunden /
            Audio nicht mehr im RAM.
        """
        session = self.get(session_id)
        if session is None:
            return None
        if session._original is None or session._restored is None:
            return None
        return session._original.copy(), session._restored.copy()

    def list_sessions(self) -> list[dict]:
        """Gibt alle gespeicherten Sessions als Liste von Dicts zurück (ohne Audio)."""
        with self._lock:
            return [s.as_dict() for s in reversed(list(self._sessions.values()))]

    def latest_session_id(self) -> str | None:
        """Gibt die zuletzt gespeicherte Session-ID zurück."""
        with self._lock:
            if not self._sessions:
                return None
            return next(reversed(self._sessions))

    def clear(self) -> None:
        """Löscht alle Sessions aus dem RAM (Sidecar-Dateien bleiben)."""
        with self._lock:
            self._sessions.clear()
        logger.debug("ABCompareManager: RAM-Zwischenspeicher geleert")

    # -------------------------------------------------------------------
    # Interne Berechnungen
    # -------------------------------------------------------------------

    @staticmethod
    def _sha256_audio(audio: np.ndarray) -> str:
        """Deterministischer SHA256-Fingerabdruck eines Audio-Arrays (16 Hex)."""
        h = hashlib.sha256()
        h.update(audio.tobytes())
        return h.hexdigest()[:16]

    @staticmethod
    def _rms_db(audio: np.ndarray) -> float:
        """RMS-Pegel in dBFS. Stille → -120 dB."""
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 1e-12:
            return -120.0
        val = 20.0 * math.log10(rms)
        return float(np.clip(val, -120.0, 0.0))

    @staticmethod
    def _spectral_similarity(
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        n_fft: int = 2048,
    ) -> float:
        """
        Spektrale Ähnlichkeit auf Log-Mel-Basis ∈ [0, 1].

        Formel:
            sim = 1 − MAE(log_mel_orig, log_mel_rest) / max_possible_mae
            max_possible_mae ≈ 10  (typischer Log-Mel-Wertebereich ca. -100..0 dB)

        Robustheit: kurzes Audio oder fehlende FFT → Fallback 0.5
        """
        try:
            # Mono-Konversion (nur Kanal 0 wenn Stereo)
            orig_m = original.flatten() if original.ndim == 1 else original[:, 0]
            rest_m = restored.flatten() if restored.ndim == 1 else restored[:, 0]

            # Gemeinsame Länge
            min_len = min(len(orig_m), len(rest_m))
            if min_len < n_fft:
                return 0.5
            orig_m = orig_m[:min_len]
            rest_m = rest_m[:min_len]

            # STFT
            hop = n_fft // 4
            n_frames = (min_len - n_fft) // hop + 1
            window = np.hanning(n_fft).astype(np.float32)

            def log_mel_approx(signal: np.ndarray) -> np.ndarray:
                """Vereinfachtes Log-Betragsspektrum (kein Mel — ausreichend für Ähnlichkeit)."""
                frames = np.stack([signal[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
                spec = np.abs(np.fft.rfft(frames, axis=-1))
                spec = np.clip(spec, 1e-8, None)
                return 20.0 * np.log10(spec)  # type: ignore[no-any-return]

            lm_orig = log_mel_approx(orig_m)
            lm_rest = log_mel_approx(rest_m)

            mae = float(np.mean(np.abs(lm_orig - lm_rest)))
            # Normierung: typischer MAE-Raum ≈ 0..20 dB
            sim = float(np.clip(1.0 - mae / 20.0, 0.0, 1.0))
            return sim

        except Exception as exc:
            logger.warning("§G23/§G93 spectral_similarity DSP-Ersatzpfad — returning 0.5: %s", exc, exc_info=True)
            return 0.5

    @staticmethod
    def _snr_estimate(original: np.ndarray, restored: np.ndarray) -> float:
        """
        Einfache SNR-Schätzung: SNR = 20·log10(RMS_signal / RMS_noise)
        wobei noise = original − restored (Differenz-Signal).

        Positiver Wert → restauriertes Audio hat weniger störende Energie.
        Gibt 0.0 zurück wenn Original zu leise für sinnvolle Schätzung.
        """
        min_len = min(len(original.flatten()), len(restored.flatten()))
        if min_len < 128:
            return 0.0
        orig_f = original.flatten()[:min_len]
        rest_f = restored.flatten()[:min_len]
        noise = orig_f - rest_f
        rms_sig = float(np.sqrt(np.mean(orig_f**2)))
        rms_noise = float(np.sqrt(np.mean(noise**2)))
        if rms_sig < 1e-12 or rms_noise < 1e-12:
            return 0.0
        snr = 20.0 * math.log10(rms_sig / rms_noise)
        return float(np.clip(snr, -60.0, 60.0))

    def _compute_diff(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> ABDiff:
        """Berechnet alle quantitativen Differenz-Metriken zwischen Original und Restauriert."""
        # Leere Arrays: numerische Metriken nicht berechenbar → sichere Standardwerte
        if original.size == 0 or restored.size == 0:
            return ABDiff(
                rms_original_db=-120.0,
                rms_restored_db=-120.0,
                rms_improvement_db=0.0,
                spectral_similarity=0.5,
                snr_estimate_db=0.0,
                peak_original=0.0,
                peak_restored=0.0,
                duration_seconds=0.0,
                n_channels=1,
            )
        rms_orig = self._rms_db(original)
        rms_rest = self._rms_db(restored)

        # Verbesserungs-Delta: negativeres RMS = weniger Rauschenergie → positives Delta
        rms_delta = rms_orig - rms_rest  # positiv wenn restauriert leiser (weniger Rauschen)

        spectral_sim = self._spectral_similarity(original, restored, sr)
        snr_est = self._snr_estimate(original, restored)

        peak_orig = float(np.max(np.abs(original)))
        peak_rest = float(np.max(np.abs(restored)))

        # Dauer: beide könnten unterschiedliche Längen haben
        duration = len(original.flatten()) / max(sr, 1)
        n_ch = original.shape[-1] if original.ndim == 2 else 1

        return ABDiff(
            rms_original_db=rms_orig,
            rms_restored_db=rms_rest,
            rms_improvement_db=rms_delta,
            spectral_similarity=spectral_sim,
            snr_estimate_db=snr_est,
            peak_original=float(np.clip(peak_orig, 0.0, 1.0)),
            peak_restored=float(np.clip(peak_rest, 0.0, 1.0)),
            duration_seconds=duration,
            n_channels=n_ch,
        )

    def _write_sidecar(self, session: ABSession) -> None:
        """
        Schreibt eine kompakte JSON-Sidecar-Datei in ~/.aurik/ab_sessions/.
        Enthält keine Audio-Daten — nur Metriken und IDs.
        """
        try:
            path = _AB_SESSION_DIR / f"{session.session_id[:8]}.json"
            payload = session.as_dict()
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.debug("AB-Sidecar Schreibfehler: %s", exc)


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (§3.2 Spec: Double-Checked Locking)
# ---------------------------------------------------------------------------

_instance: ABCompareManager | None = None
_lock = threading.Lock()


def get_ab_manager() -> ABCompareManager:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ABCompareManager()
    return _instance


# ---------------------------------------------------------------------------
# Convenience-Funktion
# ---------------------------------------------------------------------------


def store_ab_session(
    original: np.ndarray,
    restored: np.ndarray,
    sample_rate: int,
    material: str = "unknown",
    tags: list[str] | None = None,
) -> str:
    """
    Convenience-Wrapper: Speichert ein A/B-Paar und gibt die Session-ID zurück.

    Film-Analogie: Original = unbearbeiteter Rohschnitt,
                   Restored = fertig restaurierter Film.

    Args:
        original:    Original-Audio float32 [samples] oder [samples, channels]
        restored:    Restauriertes Audio (gleiche Dimensionen)
        sample_rate: Sample-Rate in Hz (muss 48000 bei Aurik-Verarbeitung sein)
        material:    Materialtyp-String z.B. "vinyl", "tape"
        tags:        Optionale Nutzer-Tags

    Returns:
        session_id — UUID-String für get_ab_manager().get(session_id)
    """
    return get_ab_manager().store(
        original=original,
        restored=restored,
        sample_rate=sample_rate,
        material=material,
        tags=tags,
    )
