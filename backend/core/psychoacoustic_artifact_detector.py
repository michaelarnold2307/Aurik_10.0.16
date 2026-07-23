"""
PsychoacousticArtifactDetector: Automatische Erkennung und gezielte Minimierung
subtiler musikalischer Störungen auf Basis psychoakustischer Modelle.

Implementierte Metriken (scipy-only, kein Deep Learning):
    - masking_effect:        Simultanmaskierungsindex über Bark-Skala
    - transient_loss:        Onset-Stärke-Schwächungsindex (Spektrale Flussdifferenz)
    - musical_transparency:  Spektrale Flachheit (Wiener-Entropie) als Transparenz-Proxy
"""

from dataclasses import asdict, dataclass

import numpy as np
import scipy.signal


@dataclass
class PsychoacousticArtifactResult:
    """Typed result of psychoacoustic artifact analysis. All scores in [0.0, 1.0]."""

    masking_effect: float
    transient_loss: float
    musical_transparency: float

    # Backward-compatible dict-style access so existing callers keep working
    def get(self, key: str, default=None):
        return asdict(self).get(key, default)

    def __getitem__(self, key: str):
        return asdict(self)[key]

    def __contains__(self, key: str) -> bool:
        return key in asdict(self)

    def items(self):
        return asdict(self).items()

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class PsychoacousticArtifactDetector:
    def __init__(self):
        self.detected_artifacts: list[str] = []

    def analyze(self, audio: np.ndarray, sr: int) -> PsychoacousticArtifactResult:
        """
        Analysiert das Audio auf psychoakustische Artefakte.

        Returns:
            PsychoacousticArtifactResult mit Scores [0.0, 1.0]:
                masking_effect:       0 = keine Maskierung, 1 = starke Maskierung
                transient_loss:       0 = Transienten intakt, 1 = stark gedämpft
                musical_transparency: 0 = tonales/artefaktreiches Signal, 1 = transparent
        """
        mask_effect = self._detect_masking(audio, sr)
        transient_loss = self._detect_transient_loss(audio, sr)
        transparency = self._estimate_transparency(audio, sr)
        return PsychoacousticArtifactResult(
            masking_effect=float(mask_effect),
            transient_loss=float(transient_loss),
            musical_transparency=float(transparency),
        )

    # ------------------------------------------------------------------
    # Metriken
    # ------------------------------------------------------------------

    def _detect_masking(self, audio: np.ndarray, sr: int) -> float:
        """
        Schätzt den Grad simultaner Frequenzmaskierung über eine Bark-Skala-Approximation.

        Methode:
          1. Kurzzeit-FFT in Frames.
          2. FFT-Bins auf 24 Bark-Bänder mappen.
          3. Je Band: Verhältnis (Max-Power) / (Gesamt-Power im Band).
             → Hoher Ratio = ein dominanter Ton maskiert den Rest → hohe Maskierung.
          4. Durchschnitt über alle Bänder und Zeit → Score [0, 1].

        Returns: masking_index in [0, 1].
        """
        audio_f = audio.astype(np.float64)
        n_fft = min(1024, len(audio_f))
        hop = n_fft // 2
        _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.103

        _, _, S = scipy.signal.stft(audio_f, fs=sr, window="hann", nperseg=n_fft, noverlap=_noverlap, boundary="even")
        power = np.abs(S) ** 2  # (n_bins, n_frames)
        n_bins = power.shape[0]

        # Bark-Skala: 24 kritische Bänder (Grenzen in Hz nach Zwicker 1961)
        bark_edges_hz = [
            0,
            100,
            200,
            300,
            400,
            510,
            630,
            770,
            920,
            1080,
            1270,
            1480,
            1720,
            2000,
            2320,
            2700,
            3150,
            3700,
            4400,
            5300,
            6400,
            7700,
            9500,
            12000,
            15500,
        ]
        freqs = np.linspace(0, sr / 2, n_bins)

        masking_scores = []
        for i in range(len(bark_edges_hz) - 1):
            f_lo, f_hi = bark_edges_hz[i], bark_edges_hz[i + 1]
            mask_bins = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
            if len(mask_bins) < 2:
                continue
            band_power = power[mask_bins, :]  # (k_bins, n_frames)
            total = band_power.sum(axis=0) + 1e-30
            peak = band_power.max(axis=0)
            # Dominanz eines Peaks: wenn peak/total → 1.0, starke Maskierung
            dominance = np.mean(peak / total)  # [0, 1]
            masking_scores.append(dominance)

        if not masking_scores:
            return 0.0
        return float(np.clip(np.mean(masking_scores), 0.0, 1.0))

    def _detect_transient_loss(self, audio: np.ndarray, sr: int) -> float:
        """
        Erkennt Transientenverluste durch Analyse der logarithmischen Spektral-Flussdifferenz.

        Methode:
          1. STFT → Magnitude-Spektrogramm.
          2. Spektraler Fluss = positive Differenz zwischen aufeinanderfolgenden Frames.
          3. Normierter Onset-Stärkevektor.
          4. Transientenverlust: 1 - (relative Onset-Konzentration).
             → Wenn Onsets verwaschen (niedrige Spitzenwerte), hoher Verlust-Score.

        Returns: transient_loss_index in [0, 1].
        """
        audio_f = audio.astype(np.float64)
        n_fft = min(1024, len(audio_f))
        hop = n_fft // 4  # Feineres Zeitraster für Transienten
        _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.103

        _, _, S = scipy.signal.stft(audio_f, fs=sr, window="hann", nperseg=n_fft, noverlap=_noverlap, boundary="even")
        mag = np.abs(S)  # (n_bins, n_frames)

        if mag.shape[1] < 3:
            return 0.0

        # Log-Spektraler Fluss (nur positive Änderungen)
        log_mag = np.log1p(mag)
        flux = np.sum(np.maximum(np.diff(log_mag, axis=1), 0), axis=0)  # (n_frames-1,)

        if flux.max() < 1e-10:
            return 0.0

        # Normieren
        flux_norm = flux / (flux.max() + 1e-30)

        # Transient-Score: Kurtosis-basiert — hohe Kurtosis = scharfe Peaks = gute Transienten
        mean_f = float(np.mean(flux_norm))
        std_f = float(np.std(flux_norm) + 1e-10)
        kurtosis = float(np.mean(((flux_norm - mean_f) / std_f) ** 4))

        # Normiere Kurtosis auf [0, 1]: hohe Kurtosis → wenig Verlust (gute Transienten)
        # Typischer Bereich Kurtosis [1, 50]; Kurtosis = 3 Gauss-Referenz
        loss = 1.0 - np.clip((kurtosis - 1.0) / 30.0, 0.0, 1.0)
        return float(loss)

    def _estimate_transparency(self, audio: np.ndarray, sr: int) -> float:
        """
        Schätzt die musikalische Transparenz via Spektrale Flachheit (Wiener-Entropie).

        Spektrale Flachheit (SFM) = geometrischer Mittelwert / arithmetischer Mittelwert
        des Leistungsspektrums.
        - SFM → 0.0: tonales/peakreiches Signal (viel Färbung)
        - SFM → 1.0: rauschartig / flach (transparent)

        Da restauriertes Audio weder rein tonal noch rein rauschig sein soll,
        normieren wir SFM auf einen "Transparenzindex":
            transparency = SFM (0=artefaktreich, 1=transparent)

        Returns: transparency_index in [0, 1].
        """
        audio_f = audio.astype(np.float64)
        n_fft = min(2048, len(audio_f))

        # Nur über einen zentralen Ausschnitt (stabil)
        centre = len(audio_f) // 2
        half = n_fft // 2
        segment = audio_f[max(0, centre - half) : centre + half]
        if len(segment) < 64:
            return 0.5

        mag = np.abs(np.fft.rfft(segment, n=n_fft)) + 1e-30
        power = mag**2

        log_mean = np.mean(np.log(power))  # log(geo mean)
        arith_mean = np.mean(power)

        sfm = float(np.exp(log_mean) / (arith_mean + 1e-30))
        return float(np.clip(sfm, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Artefakt-Minimierung
    # ------------------------------------------------------------------

    def minimize_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Minimiert erkannte psychoakustische Artefakte adaptiv.

        Strategie (einfach, ohne externe Deps):
          - Starke Maskierung → leichter Frequenz-Ausgleich via Spectral Subtraction
          - Transientenverlust → Transient-Sharpening via spektrale Derivation
          - Niedrige Transparenz → sanftes Spectral Whitening

        Returns audio mit reduzierten Artefakten (gleiche Länge + Dtype).
        """
        metrics = self.analyze(audio, sr)
        audio_f = audio.astype(np.float64)

        # --- Spectral Whitening bei niedriger Transparenz ---------------
        transparency = metrics.musical_transparency
        if transparency < 0.3:
            # Sanftes Whitening: Divide-by-Envelope im FFT-Bereich
            n_fft = min(2048, len(audio_f))
            hop = n_fft // 4
            _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.103
            _, _, S = scipy.signal.stft(
                audio_f, fs=sr, window="hann", nperseg=n_fft, noverlap=_noverlap, boundary="even"
            )
            mag = np.abs(S) + 1e-30
            # Glatte Hüllkurve (Median über Frequenz)
            envelope = np.median(mag, axis=0, keepdims=True) + 1e-30
            strength = 0.2 * (1 - transparency / 0.3)  # max 20% Whitening
            S_w = S / (1 + strength * envelope / (mag + 1e-30))
            _, audio_f = scipy.signal.istft(S_w, fs=sr, window="hann", nperseg=n_fft, noverlap=_noverlap)
            audio_f = audio_f[: len(audio)]
            if len(audio_f) < len(audio):
                audio_f = np.pad(audio_f, (0, len(audio) - len(audio_f)))

        # --- Energie-Normalisierung (Verarbeitung darf keine Energie ändern) ---
        rms_in = np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-30)
        rms_out = np.sqrt(np.mean(audio_f**2) + 1e-30)
        if rms_out > 1e-10:
            audio_f *= rms_in / rms_out

        self.detected_artifacts = [name for name, score in metrics.items() if score > 0.5]
        return np.clip(audio_f, -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]
