"""
KI-Qualitätsanalyse-Modul für Multi-Pass Processing
====================================================

Dieses Modul bewertet die Qualität von Audio-Verarbeitungsschritten
und ermöglicht die Auswahl des besten Ergebnisses aus mehreren Passes.

Nutzt echte Audio-Qualitätsmetriken:
- Spectral Clarity (FFT-basiert, Spektrale Reinheit)
- Harmonic Distortion (THD, Klirrfaktor)
- Dynamic Range (Peak vs. RMS Ratio in dB)
- Signal-to-Noise Ratio (SNR, Rauschanteil)
- Transient Preservation (Attack-Qualität)
- Phase Coherence (Stereo Phasenbeziehung)

Version: 2.0.0 - Vollständige Heuristic Implementation
Autor: AURIK System
Datum: 2026-02-13
Status: PRODUKTIONSREIF ✅
"""

import logging
import warnings
from typing import Any

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

# Suppress scipy warnings für cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


class KIQualityAnalyzer:
    """
    Heuristische Audio-Qualitätsanalyse für Multi-Pass Optimization.

    Bewertet Audio-Qualität mit 6 wissenschaftlichen Metriken:
    1. Spectral Clarity (30%) - Spektrale Reinheit, Frequenzbalance
    2. Harmonic Distortion (30%) - THD, Klirrverzerrungen
    3. Dynamic Range (20%) - Peak-to-RMS Ratio
    4. Signal-to-Noise Ratio (15%) - Nutzsignal vs. Rauschen
    5. Transient Preservation (3%) - Attack-Schärfe
    6. Phase Coherence (2%) - Stereo Phasenkorrektheit

    Score-Range: 0.0 (sehr schlecht) bis 1.0 (perfekt)
    Weltklasse-Audio: ≥ 0.85
    Akzeptabel: ≥ 0.70
    Problematisch: < 0.60
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        # Frequenzbänder für Spectral Clarity Analyse
        self.freq_bands = {
            "sub_bass": (20, 60),  # Sub-Bass
            "bass": (60, 250),  # Bass
            "low_mid": (250, 500),  # Untere Mitten
            "mid": (500, 2000),  # Mitten
            "high_mid": (2000, 4000),  # Obere Mitten
            "presence": (4000, 8000),  # Präsenz
            "brilliance": (8000, 20000),  # Brillanz
        }

    def analyze_chain(
        self,
        audio: np.ndarray,
        sr: int,
        processing_chain: list | None = None,
        forensic_analysis: dict | None = None,
    ) -> float:
        """
        Hauptmethode: Bewertet Audio-Qualität mit allen Metriken.

        Args:
            audio: Audio signal (mono oder stereo)
            sr: Sample rate
            processing_chain: Optional Liste von Processing-Modulen (für Fallback)
            forensic_analysis: Optional forensische Analyse (für Fallback)

        Returns:
            Quality score (0.0 - 1.0, höher ist besser)
        """
        try:
            # Audio validation
            if audio is None or len(audio) == 0:
                self.logger.warning("Empty audio in analyze_chain")
                return 0.0

            # Convert zu Mono für Analyse (falls Stereo)
            if audio.ndim == 2:
                audio_mono = np.mean(audio, axis=1)
                stereo = True
            else:
                audio_mono = audio
                stereo = False

            # 1. Spectral Clarity (30%)
            spectral_clarity = self._measure_spectral_clarity(audio_mono, sr)

            # 2. Harmonic Distortion - inverted (30%)
            thd = self._measure_thd(audio_mono, sr)
            thd_score = 1.0 - np.clip(thd, 0.0, 1.0)

            # 3. Dynamic Range (20%)
            dr_score = self._measure_dynamic_range(audio_mono)

            # 4. Signal-to-Noise Ratio (15%)
            snr_score = self._measure_snr(audio_mono, sr)

            # 5. Transient Preservation (3%)
            transient_score = self._measure_transient_quality(audio_mono, sr)

            # 6. Phase Coherence (2% - nur wenn Stereo)
            if stereo:
                phase_score = self._measure_phase_coherence(audio)
            else:
                phase_score = 1.0  # Perfect für Mono

            # Gewichtete Kombination
            quality_score = (
                0.30 * spectral_clarity
                + 0.30 * thd_score
                + 0.20 * dr_score
                + 0.15 * snr_score
                + 0.03 * transient_score
                + 0.02 * phase_score
            )

            final_score = np.clip(quality_score, 0.0, 1.0)

            self.logger.debug(
                f"Quality Analysis: SC={spectral_clarity:.3f}, THD={thd_score:.3f}, "
                f"DR={dr_score:.3f}, SNR={snr_score:.3f}, Trans={transient_score:.3f}, "
                f"Phase={phase_score:.3f} → FINAL={final_score:.3f}"
            )

            return float(final_score)

        except Exception as e:
            logger.error("KI-Quality Chain Analyse fehlgeschlagen: %s", e, exc_info=True)
            # Fallback zu alter Logik wenn vorhanden
            if processing_chain is not None and forensic_analysis is not None:
                return self._fallback_chain_analysis(processing_chain, forensic_analysis)
            return 0.6  # Neutral fallback

    def _fallback_chain_analysis(self, processing_chain: list, forensic_analysis: dict) -> float:
        """Legacy fallback für Kompatibilität."""
        try:
            if not processing_chain:
                return 0.5

            # Basis-Score: Je kürzer die Chain, desto höher der Score
            chain_score = max(0.3, 1.0 - (len(processing_chain) * 0.05))

            # Forensic confidence berücksichtigen
            forensic_confidence = forensic_analysis.get("medium_confidence", 0.5)

            # Kombinierter Score
            quality_score = (chain_score * 0.7) + (forensic_confidence * 0.3)

            return np.clip(quality_score, 0.0, 1.0)  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning("Ersatzpfad chain Analyse fehlgeschlagen: %s", e)
            return 0.5

    # ============================================================================
    # MEASUREMENT METHODS - Echte Audio-Qualitätsmetriken
    # ============================================================================

    def _measure_spectral_clarity(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst spektrale Klarheit via FFT-Analyse.

        Bewertet:
        - Frequenzbalance über alle Bänder
        - Spektrale Flatness (Rauschen vs. Ton)
        - Energie-Verteilung

        Returns:
            Score 0.0-1.0 (1.0 = perfekte Klarheit)
        """
        try:
            # FFT-Analyse
            n_fft = 4096
            if len(audio) < n_fft:
                n_fft = len(audio)
            window = signal.get_window("hann", n_fft)

            # Compute Power Spectral Density
            f, pxx = signal.welch(audio, sr, window=window, nperseg=n_fft, noverlap=n_fft // 2)

            # Spektrale Flatness (Wiener Entropy)
            # High flatness = Noise-like = BAD
            # Low flatness = Tonal = GOOD (für Musik)
            pxx_safe = pxx + 1e-10
            geometric_mean = np.exp(np.mean(np.log(pxx_safe)))
            arithmetic_mean = np.mean(pxx_safe)
            spectral_flatness = geometric_mean / arithmetic_mean

            # Invert: Low flatness = High score
            flatness_score = 1.0 - np.clip(spectral_flatness, 0.0, 1.0)

            # Frequenzbalance: Energie sollte gut verteilt sein
            band_energies = []
            for band_name, (f_low, f_high) in self.freq_bands.items():
                idx = np.where((f >= f_low) & (f <= f_high))[0]
                if len(idx) > 0:
                    band_energy: float = float(np.sum(pxx[idx]))
                    band_energies.append(band_energy)

            if len(band_energies) > 0:
                # Standard deviation von Log-Energien (niedrig = gut balanciert)
                log_energies = np.log10(np.array(band_energies) + 1e-10)
                balance_std = np.std(log_energies)
                # Normalisiere: std von 0-3 dB ist gut
                balance_score = np.clip(1.0 - (balance_std / 3.0), 0.0, 1.0)
            else:
                balance_score = 0.5

            # Kombiniere beide Metriken
            clarity_score = (0.6 * flatness_score) + (0.4 * balance_score)

            return float(np.clip(clarity_score, 0.0, 1.0))

        except Exception as e:
            logger.warning("Spectral clarity measurement fehlgeschlagen: %s", e)
            return 0.7  # Neutral fallback

    def _measure_thd(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Total Harmonic Distortion (THD).

        THD = Sqrt(Sum(Harmonic_Powers)) / Fundamental_Power

        Returns:
            THD value 0.0-1.0+ (0.0 = keine Verzerrung, höher = mehr Verzerrung)
            Typische Werte:
            - <0.01 (1%): Exzellent
            - 0.01-0.05 (1-5%): Gut
            - >0.10 (10%): Problematisch
        """
        try:
            # FFT-Analyse mit hoher Auflösung
            n_fft = 8192
            audio_windowed = audio * signal.get_window("hann", len(audio))
            spectrum = np.abs(np.fft.fft(audio_windowed, n=n_fft))
            freqs = np.fft.fftfreq(n_fft, 1 / sr)

            # Nur positpve Frequenzen
            pos_mask = freqs > 0
            spectrum = spectrum[pos_mask]
            freqs = freqs[pos_mask]

            # Finde Fundamental (stärkste Frequenz in 50-2000 Hz)
            fundamental_range = (freqs >= 50) & (freqs <= 2000)
            if not np.any(fundamental_range):
                return 0.05  # Neutral fallback

            fundamental_idx = np.argmax(spectrum[fundamental_range])
            fundamental_freq = freqs[fundamental_range][fundamental_idx]
            fundamental_power = spectrum[fundamental_range][fundamental_idx] ** 2

            # Finde Harmonics (2f, 3f, 4f, 5f, 6f)
            harmonic_power_sum = 0.0
            for n in range(2, 7):  # 2nd bis 6th harmonic
                harmonic_freq = fundamental_freq * n
                # Finde nächste Frequenz
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                if freqs[harmonic_idx] < sr / 2:  # Unter Nyquist
                    harmonic_power_sum += spectrum[harmonic_idx] ** 2

            # THD berechnen
            thd = np.sqrt(harmonic_power_sum) / np.sqrt(fundamental_power) if fundamental_power > 0 else 0.5

            # Normalisiere: >0.2 (20%) wird auf 1.0 gecapped
            thd_normalized = np.clip(thd / 0.2, 0.0, 1.0)

            return float(thd_normalized)

        except Exception as e:
            logger.warning("THD measurement fehlgeschlagen: %s", e)
            return 0.05  # Low distortion fallback

    def _measure_dynamic_range(self, audio: np.ndarray) -> float:
        """
        Misst Dynamic Range als Peak-to-RMS Ratio.

        DR_dB = 20 * log10(Peak / RMS)

        Typische Werte:
        - 40+ dB: Exzellent (klassische Musik)
        - 20-40 dB: Gut (Rock, Pop)
        - 10-20 dB: Komprimiert (Modern Pop)
        - <10 dB: Sehr komprimiert (Loudness War)

        Returns:
            Score 0.0-1.0 (1.0 = exzellenter Dynamic Range)
        """
        try:
            # Peak
            peak = np.abs(audio).max()

            # RMS
            rms = np.sqrt(np.mean(audio**2))

            if peak == 0 or rms == 0:
                return 0.1  # Keine Dynamik

            # Dynamik in dB
            dr_db = 20 * np.log10(peak / (rms + 1e-10))

            # Normalisiere:
            # 60 dB = 1.0 (perfekt)
            # 30 dB = 0.5 (akzeptabel)
            # 0 dB = 0.0 (keine Dynamik)
            dr_score = np.clip(dr_db / 60.0, 0.0, 1.0)

            return float(dr_score)

        except Exception as e:
            logger.warning("Dynamic range measurement fehlgeschlagen: %s", e)
            return 0.5  # Medium fallback

    def _measure_snr(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Signal-to-Noise Ratio via Noise Floor Schätzung.

        Methode: Vergleiche Signal-RMS mit Noise Floor (leise Abschnitte)

        Returns:
            Score 0.0-1.0 (1.0 = sehr hohes SNR, sehr sauber)
        """
        try:
            # Finde leise Abschnitte (potentiell Noise)
            # Frame-based analysis
            frame_length = sr // 10  # 100ms frames
            n_frames = len(audio) // frame_length

            if n_frames < 2:
                return 0.7  # Zu kurz für Analyse

            frame_rms = []
            for i in range(n_frames):
                start = i * frame_length
                end = start + frame_length
                frame = audio[start:end]
                frame_rms.append(np.sqrt(np.mean(frame**2)))

            _frame_rms_arr = np.array(frame_rms)

            # Noise Floor = 20tes Perzentil (leise Frames)
            noise_floor = np.percentile(_frame_rms_arr, 20)

            # Signal Level = 80tes Perzentil (laute Frames)
            signal_level = np.percentile(_frame_rms_arr, 80)

            if noise_floor == 0 or signal_level == 0:
                return 0.7

            # SNR in dB
            snr_db = 20 * np.log10(signal_level / (noise_floor + 1e-10))

            # Normalisiere:
            # 60+ dB = 1.0 (perfekt, studio-quality)
            # 40 dB = 0.67 (gut)
            # 20 dB = 0.33 (akzeptabel)
            # 0 dB = 0.0 (sehr verrauscht)
            snr_score = np.clip(snr_db / 60.0, 0.0, 1.0)

            return float(snr_score)

        except Exception as e:
            logger.warning("SNR measurement fehlgeschlagen: %s", e)
            return 0.7  # Good fallback

    def _measure_transient_quality(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Transient-Qualität (Attack-Schärfe).

        Transients = schnelle Lautstärkeänderungen (Schlagzeug, Attacks)
        Gute Qualität = steile Flanken, gut erhaltene Transienten

        Returns:
            Score 0.0-1.0 (1.0 = perfekte Transient-Erhaltung)
        """
        try:
            # Envelope via Hilbert Transform
            analytic_signal: np.ndarray = signal.hilbert(audio)  # type: ignore[assignment]
            envelope = np.abs(analytic_signal)

            # Finde Transienten via Envelope-Gradient
            envelope_diff = np.diff(envelope)

            # Attack-Steilheit (hohe positive Gradienten)
            attack_threshold = np.percentile(np.abs(envelope_diff), 95)
            strong_attacks = envelope_diff > attack_threshold

            if not np.any(strong_attacks):
                return 0.7  # Keine Transienten gefunden (OK für manche Musik)

            # Durchschnittliche Attack-Steilheit
            attack_slopes = envelope_diff[strong_attacks]
            mean_attack_slope = np.mean(attack_slopes)

            # Normalisiere: Higher slope = better preservation
            # Typische Werte: 0.001 - 0.1
            transient_score = np.clip(mean_attack_slope * 10, 0.0, 1.0)

            return float(transient_score)

        except Exception as e:
            logger.warning("Transient quality measurement fehlgeschlagen: %s", e)
            return 0.75  # Good fallback

    def _measure_phase_coherence(self, audio_stereo: np.ndarray) -> float:
        """
        Misst Phase Coherence für Stereo-Audio.

        Phase Correlation = Korrelation zwischen L und R Kanälen
        +1.0 = Perfekt in Phase (Mono)
        0.0 = Unkorreliert (breites Stereo)
        -1.0 = Perfekt außer Phase (phasige Probleme)

        Optimal für Musik: 0.6 - 0.9 (gute Stereobreite ohne Phasenprobleme)

        Returns:
            Score 0.0-1.0 (1.0 = optimal für Musik)
        """
        try:
            if audio_stereo.ndim != 2 or audio_stereo.shape[0] < 2:
                return 1.0  # Mono = perfekt

            left = audio_stereo[0]
            right = audio_stereo[1]

            # Correlation — guard against RuntimeWarning on near-constant/silent channels
            std_l = float(np.std(left))
            std_r = float(np.std(right))
            if std_l < 1e-10 or std_r < 1e-10:
                correlation = 1.0 if (std_l < 1e-10 and std_r < 1e-10) else 0.0
            else:
                _la = left - left.mean()
                _ra = right - right.mean()
                _nl = float(np.linalg.norm(_la))
                _nr = float(np.linalg.norm(_ra))
                _c = float(np.dot(_la, _ra) / (_nl * _nr + 1e-10))
                correlation = _c if np.isfinite(_c) else 1.0

            # Bewerte: Optimal ist 0.6 - 0.9
            if 0.6 <= correlation <= 0.9:
                score = 1.0  # Optimal
            elif 0.4 <= correlation < 0.6 or 0.9 < correlation <= 1.0:
                score = 0.8  # Gut
            elif 0.0 <= correlation < 0.4:
                score = 0.6  # Akzeptabel (sehr breit)
            else:  # Negative correlation = Phasenprobleme
                score = 0.3  # Problematisch

            return float(score)

        except Exception as e:
            logger.warning("Verarbeitungsschritt coherence measurement fehlgeschlagen: %s", e)
            return 0.8  # Good fallback

    # ============================================================================
    # LEGACY METHODS - Für Kompatibilität mit bestehendem Code
    # ============================================================================

    def analyze_digital_metrics(self, digital_metrics: dict, audio: np.ndarray) -> float:
        """
        Bewertet Digital-Restoration Ergebnisse.

        Args:
            digital_metrics: Metriken vom DigitalRestorationSpecialist
            audio: Verarbeitetes Audio

        Returns:
            Quality score (0.0 - 1.0, höher ist besser)
        """
        try:
            score = 0.7  # Basis-Score

            # Packet Loss Repair verbessert Qualität
            if digital_metrics.get("packet_loss", {}).get("gaps_found", 0) > 0:
                score += 0.1

            # Jitter Correction verbessert Qualität
            if digital_metrics.get("jitter", {}).get("jitter_detected", False):
                score += 0.1

            # Codec Artifact Removal verbessert Qualität
            if digital_metrics.get("codec_artifacts", {}).get("pre_echo_detected", False):
                score += 0.1

            return np.clip(score, 0.0, 1.0)  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning("KI-Quality Digital Analyse fehlgeschlagen: %s", e)
            return 0.7  # Positive fallback

    def analyze_multi_track_metrics(self, mt_metrics: dict, audio: np.ndarray) -> float:
        """
        Bewertet Multi-Track Enhancement Ergebnisse.

        Args:
            mt_metrics: Metriken vom MultiTrackSpecialist
            audio: Verarbeitetes Audio

        Returns:
            Quality score (0.0 - 1.0, höher ist besser)
        """
        try:
            score = 0.7  # Basis-Score

            # Time Alignment verbessert Qualität
            if mt_metrics.get("time_alignment", {}).get("alignment_applied", False):
                score += 0.1

            # Phase Alignment verbessert Qualität
            if mt_metrics.get("phase_alignment", {}).get("correction_applied", False):
                score += 0.1

            # Stereo Balance verbessert Qualität
            if mt_metrics.get("balance", {}).get("correction_applied", False):
                score += 0.05

            return np.clip(score, 0.0, 1.0)  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning("KI-Quality Multi-Track Analyse fehlgeschlagen: %s", e)
            return 0.7  # Positive fallback

    def analyze_mono_to_stereo_metrics(self, metrics: dict, audio: np.ndarray) -> float:
        """
        Bewertet Mono-to-Stereo Upmixing.

        Args:
            metrics: Mono-to-Stereo Metriken
            audio: Verarbeitetes Audio

        Returns:
            Quality score (0.0 - 1.0, höher ist besser)
        """
        try:
            # Phase Correlation sollte zwischen 0.7 - 0.9 liegen (optimal)
            phase_corr = metrics.get("phase_correlation", 0.8)
            if 0.7 <= phase_corr <= 0.9:
                score = 0.9
            elif 0.5 <= phase_corr < 0.7 or 0.9 < phase_corr <= 1.0:
                score = 0.7
            else:
                score = 0.5

            # Mono Compatibility Check
            if metrics.get("mono_compatible", True):
                score += 0.05

            return np.clip(score, 0.0, 1.0)  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning("KI-Quality Mono-to-Stereo Analyse fehlgeschlagen: %s", e)
            return 0.75

    def analyze_spectral_metrics(self, metrics: dict, audio: np.ndarray) -> float:
        """
        Generische spektrale Qualitätsbewertung.

        Args:
            metrics: Spektrale Metriken (spectral_clarity, harmonic_distortion, etc.)
            audio: Verarbeitetes Audio

        Returns:
            Quality score (0.0 - 1.0, höher ist besser)
        """
        try:
            # Nutze spectral_clarity wenn verfügbar
            spectral_clarity = metrics.get("spectral_clarity", 0.7)

            # Nutze harmonic_distortion wenn verfügbar (niedrig ist gut)
            harmonic_distortion = metrics.get("harmonic_distortion", 0.1)
            distortion_score = 1.0 - np.clip(harmonic_distortion, 0.0, 1.0)

            # Kombiniere Metriken
            quality_score = (spectral_clarity * 0.6) + (distortion_score * 0.4)

            return np.clip(quality_score, 0.0, 1.0)  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning("KI-Quality Spectral Analyse fehlgeschlagen: %s", e)
            return 0.7

    def analyze_audio_quality(self, audio: np.ndarray, sr: int, metrics: dict | None = None) -> float:
        """
        Universelle Audio-Qualitätsbewertung.

        NEUE VERSION: Nutzt vollständige analyze_chain Implementierung!

        Args:
            audio: Audio signal
            sr: Sample rate
            metrics: Optional zusätzliche Metriken (für legacy compatibility)

        Returns:
            Quality score (0.0 - 1.0, höher ist besser)
        """
        try:
            # Nutze die vollständige analyze_chain Methode
            return self.analyze_chain(audio, sr, processing_chain=None, forensic_analysis=None)

        except Exception as e:
            logger.warning("KI-Quality Audio Analyse fehlgeschlagen: %s", e)
            return 0.6

    def analyze_multitrack_metrics(self, metrics: dict[str, Any], audio: np.ndarray | None = None) -> float:
        """
        Bewertet Multi-Track Enhancement Qualität basierend auf Metriken.

        Args:
            metrics: Dict mit Multi-Track Metriken (time_alignment, phase_alignment, etc.)
            audio: Optional audio für zusätzliche Analyse

        Returns:
            Quality score (0.0-1.0)
        """
        try:
            score = 1.0  # Start mit perfekt

            # Time Alignment Quality
            if "time_alignment" in metrics:
                ta = metrics["time_alignment"]
                if ta.get("alignment_applied", False):
                    delay_ms = abs(ta.get("delay_ms", 0.0))
                    correlation = ta.get("correlation", 1.0)
                    # Penalty für große Delays (> 1ms)
                    if delay_ms > 1.0:
                        score *= 1.0 - min(0.2, delay_ms / 10.0)
                    # Bonus für hohe Korrelation
                    score *= 0.7 + 0.3 * correlation

            # Phase Alignment Quality
            if "phase_alignment" in metrics:
                pa = metrics["phase_alignment"]
                phase_diff = abs(pa.get("phase_diff_degrees", 0.0))
                # Penalty für große Phase-Differenzen (> 15°)
                if phase_diff > 15.0:
                    score *= 1.0 - min(0.3, phase_diff / 180.0)

            # Comb Filtering Detection
            if "comb_filter" in metrics:
                cf = metrics["comb_filter"]
                if cf.get("correction_applied", False):
                    notches = cf.get("notches_detected", 0)
                    # Bonus für Comb Filter Entfernung
                    score *= 1.0 + min(0.1, notches / 100.0)

            # Stereo Balance Quality
            if "stereo_balance" in metrics:
                sb = metrics["stereo_balance"]
                if sb.get("correction_applied", False):
                    imbalance_db = abs(sb.get("imbalance_db", 0.0))
                    # Bonus für Balance-Korrektur
                    if imbalance_db > 1.0:
                        score *= 1.0 + min(0.1, imbalance_db / 10.0)

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            logger.error("Multitrack-Metrik-Analyse fehlgeschlagen: %s", e)
            return 0.8  # Conservative fallback

    def analyze_tape_metrics(self, metrics: dict[str, Any], audio: np.ndarray | None = None) -> float:
        """
        Bewertet Tape Defect Restoration Qualität.

        Args:
            metrics: Dict mit Tape Metriken (azimuth, print_through)
            audio: Optional audio für zusätzliche Analyse

        Returns:
            Quality score (0.0-1.0)
        """
        try:
            score = 1.0

            # Azimuth Correction Quality
            if "azimuth" in metrics:
                az = metrics["azimuth"]
                if az.get("correction_applied", False):
                    phase_error = abs(az.get("max_phase_error_degrees", 0.0))
                    # Bonus für Azimuth-Korrektur
                    if phase_error > 5.0:
                        score *= 1.0 + min(0.15, phase_error / 45.0)

            # Print-Through Removal Quality
            if "print_through" in metrics:
                pt = metrics["print_through"]
                if pt.get("removal_applied", False):
                    echo_db = pt.get("post_echo_attenuation_db", 0.0)
                    # Bonus für Echo-Entfernung
                    if echo_db < -40.0:  # Gute Dämpfung
                        score *= 1.1

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            logger.error("Tape-Metrik-Analyse fehlgeschlagen: %s", e)
            return 0.85  # Conservative fallback


# Factory function for easy import
def create_ki_quality_analyzer() -> KIQualityAnalyzer:
    """Erstellt eine neue KIQualityAnalyzer Instanz."""
    return KIQualityAnalyzer()
