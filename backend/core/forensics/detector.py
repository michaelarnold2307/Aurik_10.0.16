"""
forensics/detector.py
Multi-Layer Forensik-Engine
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from backend.core.forensics.signatures import MEDIA_SIGNATURES, MediaType

logger = logging.getLogger(__name__)


@dataclass
class ForensicEvidence:
    """Einzelner forensischer Befund."""

    feature: str
    detected_value: float
    expected_range: tuple[float, float]
    confidence: float
    supports_media: list[MediaType]
    contradicts_media: list[MediaType]
    description: str


@dataclass
class MediaHypothesis:
    """Hypothese über ein Medium."""

    media_type: MediaType
    confidence: float
    evidence: list[ForensicEvidence]

    # Transfer Detection
    is_transfer: bool = False
    original_media: MediaType | None = None
    intermediate_media: list[MediaType] = field(default_factory=list)


@dataclass
class ForensicReport:
    """Vollständiger forensischer Bericht."""

    # Primäre Erkennung
    primary_media: MediaType
    primary_confidence: float

    # Alternative Hypothesen
    alternatives: list[MediaHypothesis]

    # Transfer-Analyse
    transfer_detected: bool
    transfer_chain: list[MediaType]

    # Alle Beweise
    all_evidence: list[ForensicEvidence]

    # Zusammenfassung
    summary: str


class MediaForensicsEngine:
    def _map_to_main_type(self, media_type):
        # Mapping von detaillierten Typen auf Haupttypen
        from backend.core.forensics.signatures import MediaType

        # Haupttypen direkt zurückgeben
        if media_type in [
            MediaType.VINYL,
            MediaType.TAPE,
            MediaType.CASSETTE,
            MediaType.CD,
            MediaType.DIGITAL_NATIVE,
            MediaType.RADIO_BROADCAST,
            MediaType.UNKNOWN,
        ]:
            return media_type

        vinyl_types = [
            MediaType.VINYL_LP_MONO,
            MediaType.VINYL_LP_STEREO,
            MediaType.VINYL_LP_QUAD,
            MediaType.VINYL_45_MONO,
            MediaType.VINYL_45_STEREO,
            MediaType.VINYL_DIRECT_TO_DISC,
            MediaType.FLEXI_DISC,
            MediaType.CYLINDER_EDISON,
            MediaType.CYLINDER_PATHE,
            MediaType.SHELLAC_ACOUSTIC,
            MediaType.SHELLAC_ELECTRIC,
        ]
        tape_types = [
            MediaType.TAPE_30IPS,
            MediaType.TAPE_15IPS,
            MediaType.TAPE_7_5IPS,
            MediaType.TAPE_3_75IPS,
            MediaType.TAPE_1_875IPS,
            MediaType.WIRE_RECORDING,
            MediaType.EIGHT_TRACK,
            MediaType.ELCASET,
            MediaType.MICROCASSETTE,
            MediaType.DCC,
            MediaType.DAT_48K,
            MediaType.DAT_44K,
            MediaType.DAT_32K,
            MediaType.ADAT,
        ]
        cassette_types = [
            MediaType.CASSETTE_TYPE_I,
            MediaType.CASSETTE_TYPE_II,
            MediaType.CASSETTE_TYPE_IV,
            MediaType.CASSETTE_DOLBY_B,
            MediaType.CASSETTE_DOLBY_C,
            MediaType.CASSETTE_DOLBY_S,
            MediaType.CASSETTE_DBX,
            MediaType.MINIDISC,
            MediaType.MINIDISC_HIMD,
        ]
        cd_types = [
            MediaType.CD_STANDARD,
            MediaType.CD_HDCD,
            MediaType.DVD_AUDIO,
            MediaType.SACD_DSD,
            MediaType.HIRES_PCM,
            MediaType.LASERDISC,
            MediaType.VHS_LINEAR,
            MediaType.VHS_HIFI,
            MediaType.BETAMAX,
            MediaType.OPTICAL_MONO,
            MediaType.OPTICAL_STEREO,
            MediaType.DOLBY_STEREO,
            MediaType.DOLBY_SR,
        ]
        digital_types = [
            MediaType.MP3_128,
            MediaType.MP3_192,
            MediaType.MP3_256,
            MediaType.MP3_320,
            MediaType.MP3_VBR,
            MediaType.AAC_128,
            MediaType.AAC_256,
            MediaType.OGG_VORBIS,
            MediaType.WMA,
            MediaType.ATRAC_SP,
            MediaType.ATRAC_LP2,
            MediaType.ATRAC_LP4,
            MediaType.OPUS,
            MediaType.AC3,
            MediaType.DTS,
        ]
        radio_types = [
            MediaType.AM_MW,
            MediaType.AM_SW,
            MediaType.FM_MONO,
            MediaType.FM_STEREO,
            MediaType.DAB,
            MediaType.DAB_PLUS,
            MediaType.SATELLITE_RADIO,
            MediaType.INTERNET_STREAM,
            MediaType.PSTN,
            MediaType.GSM,
            MediaType.VOIP,
        ]

        if media_type in vinyl_types:
            return MediaType.VINYL
        elif media_type in tape_types:
            return MediaType.TAPE
        elif media_type in cassette_types:
            # CAS Confidence Score erhöhen
            self.evidence.append(
                ForensicEvidence(
                    feature="cassette_detected",
                    detected_value=1.0,
                    expected_range=(0.8, 1.0),
                    confidence=0.95,
                    supports_media=[MediaType.CASSETTE],
                    contradicts_media=[],
                    description="Kassette erkannt: CAS erhöht",
                )
            )
            return MediaType.CASSETTE
        elif media_type in cd_types:
            return MediaType.CD
        elif media_type in digital_types:
            return MediaType.DIGITAL_NATIVE
        elif media_type in radio_types:
            return MediaType.RADIO_BROADCAST
        else:
            return MediaType.UNKNOWN

    # Multi-Layer Forensik-Engine für Tonträger-Erkennung.
    # Analysiert:
    # 1. Spektrale Charakteristik (Bandbreite, Cutoffs)
    # 2. Rausch-Fingerabdruck (PSD, Spektrale Form)
    # 3. Artefakt-Detektion (Clicks, Flutter, Codec)
    # 4. Dynamik-Analyse
    # 5. Stereo-Charakteristik
    # 6. Zeitliche Marker

    def __init__(self):
        self.signatures = MEDIA_SIGNATURES
        self.evidence = []

    def _analyze_spectral(self, audio, sr):
        # Analyse: Bandbreite, Roll-off, spektrale Energie
        from scipy.signal import welch

        f, Pxx = welch(audio, sr, nperseg=4096)
        f[np.argmax(Pxx)]
        rolloff = f[np.where(np.cumsum(Pxx) >= 0.95 * np.sum(Pxx))[0][0]]
        evidence = []
        # Vinyl/Tape: Bandbegrenzung < 20kHz, Digital: bis 22kHz
        if rolloff < 16000:
            evidence.append(
                ForensicEvidence(
                    feature="spectral_rolloff",
                    detected_value=rolloff,
                    expected_range=(0, 16000),
                    confidence=0.7,
                    supports_media=[MediaType.VINYL_LP_STEREO, MediaType.TAPE_7_5IPS],
                    contradicts_media=[MediaType.DIGITAL_NATIVE],
                    description="Rolloff < 16kHz typisch für analoge Medien",
                )
            )
        else:
            evidence.append(
                ForensicEvidence(
                    feature="spectral_rolloff",
                    detected_value=rolloff,
                    expected_range=(16000, 24000),
                    confidence=0.8,
                    supports_media=[MediaType.DIGITAL_NATIVE],
                    contradicts_media=[MediaType.VINYL_LP_STEREO, MediaType.TAPE_7_5IPS],
                    description="Rolloff > 16kHz typisch für digitale Medien",
                )
            )
        return evidence

    def _analyze_noise(self, audio, sr):
        # Analyse: Rauschpegel (Noise Floor)
        rms = np.sqrt(np.mean(audio**2))
        noise_floor = 20 * np.log10(rms + 1e-10)
        evidence = []
        if noise_floor < -40:
            evidence.append(
                ForensicEvidence(
                    feature="noise_floor",
                    detected_value=noise_floor,
                    expected_range=(-80, -40),
                    confidence=0.7,
                    supports_media=[MediaType.TAPE_7_5IPS, MediaType.VINYL_LP_STEREO],
                    contradicts_media=[MediaType.DIGITAL_NATIVE],
                    description="Niedriger Rauschpegel typisch für analoge Medien",
                )
            )
        else:
            evidence.append(
                ForensicEvidence(
                    feature="noise_floor",
                    detected_value=noise_floor,
                    expected_range=(-40, 0),
                    confidence=0.8,
                    supports_media=[MediaType.DIGITAL_NATIVE],
                    contradicts_media=[MediaType.TAPE_7_5IPS, MediaType.VINYL_LP_STEREO],
                    description="Hoher Rauschabstand typisch für digitale Medien",
                )
            )
        return evidence

    def _analyze_artifacts(self, audio, sr):
        # Analyse: Codec-Artefakte (z.B. MP3-Blockartefakte)
        # Zweite Ableitung des Signals: hohe Werte deuten auf harte Diskontinuitäten hin
        blockiness = np.mean(np.abs(np.diff(audio, n=2)))
        evidence = []
        if blockiness > 0.1:
            evidence.append(
                ForensicEvidence(
                    feature="blockiness",
                    detected_value=blockiness,  # type: ignore[arg-type]
                    expected_range=(0.1, 1.0),
                    confidence=0.7,
                    supports_media=[MediaType.MP3_128, MediaType.DIGITAL_NATIVE],
                    contradicts_media=[MediaType.VINYL_LP_STEREO, MediaType.TAPE_7_5IPS],
                    description="Blockartefakte typisch für verlustbehaftete Codecs",
                )
            )
        return evidence

    def _analyze_dynamics(self, audio, sr):
        """Dynamik-Analyse: Crest-Factor und Loudness-Range."""
        evidence = []
        try:
            rms = float(np.sqrt(np.mean(audio**2))) + 1e-12
            peak = float(np.max(np.abs(audio)))
            crest_db = float(20 * np.log10(peak / rms))
            # Hoher Crest-Factor: analoges Medium (wenig Dynamikkompression)
            if crest_db > 18:
                evidence.append(
                    ForensicEvidence(
                        feature="crest_factor_db",
                        detected_value=crest_db,
                        expected_range=(18, 40),
                        confidence=0.65,
                        supports_media=[MediaType.VINYL_LP_STEREO, MediaType.TAPE_7_5IPS],
                        contradicts_media=[MediaType.MP3_128],
                        description=f"Hoher Crest-Factor ({crest_db:.1f} dB) deutet auf analoges Medium hin",
                    )
                )
            elif crest_db < 10:
                evidence.append(
                    ForensicEvidence(
                        feature="crest_factor_db",
                        detected_value=crest_db,
                        expected_range=(0, 10),
                        confidence=0.6,
                        supports_media=[MediaType.MP3_128, MediaType.DIGITAL_NATIVE],
                        contradicts_media=[MediaType.TAPE_7_5IPS, MediaType.VINYL_LP_STEREO],
                        description=f"Niedriger Crest-Factor ({crest_db:.1f} dB) deutet auf starke Dynamikkompression hin",
                    )
                )
        except Exception:
            logger.debug("ForensicsEngine: dynamics Analyse fehlgeschlagen", exc_info=True)
        return evidence

    def _analyze_stereo(self, audio, sr):
        """Stereo-Analyse: M/S-Korrelation und Stereobreite."""
        evidence: list[ForensicEvidence] = []
        try:
            if audio.ndim < 2 or audio.shape[0] < 2:
                return evidence
            L, R = audio[0].astype(np.float64), audio[1].astype(np.float64)
            # Korrelation L/R
            corr = float(np.dot(L, R) / (np.linalg.norm(L) * np.linalg.norm(R) + 1e-12))
            if corr > 0.98:
                evidence.append(
                    ForensicEvidence(
                        feature="stereo_correlation",
                        detected_value=corr,
                        expected_range=(0.98, 1.0),
                        confidence=0.75,
                        supports_media=[MediaType.CASSETTE_TYPE_I],
                        contradicts_media=[MediaType.VINYL_LP_STEREO],
                        description="Sehr hohe L/R-Korrelation: quasi-mono (Kassette/Mono-Transfer)",
                    )
                )
            elif corr < 0.5:
                evidence.append(
                    ForensicEvidence(
                        feature="stereo_correlation",
                        detected_value=corr,
                        expected_range=(0.0, 0.5),
                        confidence=0.65,
                        supports_media=[MediaType.VINYL_LP_STEREO, MediaType.DIGITAL_NATIVE],
                        contradicts_media=[MediaType.CASSETTE_TYPE_I],
                        description=f"Breites Stereobild (Korrelation {corr:.2f})",
                    )
                )
        except Exception:
            logger.debug("ForensicsEngine: stereo Analyse fehlgeschlagen", exc_info=True)
        return evidence

    def _analyze_codecs(self, audio, sr):
        """Codec-Analyse: HF-Rolloff bei typischen MP3/AAC-Grenzfrequenzen."""
        evidence = []
        try:
            n = min(len(audio) if audio.ndim == 1 else audio.shape[-1], 8192)
            y = (audio.flatten() if audio.ndim > 1 else audio)[:n].astype(np.float64)
            mag = np.abs(np.fft.rfft(y * np.hanning(n), n=n))
            freqs = np.fft.rfftfreq(n, 1.0 / sr)
            # MP3 128 kbps: HF-Cutoff typisch bei ~16 kHz
            hf_mask_16 = freqs > 16000
            hf_mask_all = freqs > 1000
            hf_ratio = float(np.sum(mag[hf_mask_16] ** 2) / (np.sum(mag[hf_mask_all] ** 2) + 1e-12))
            if hf_ratio < 0.01 and sr >= 44100:
                evidence.append(
                    ForensicEvidence(
                        feature="hf_energy_16k",
                        detected_value=hf_ratio,
                        expected_range=(0.0, 0.01),
                        confidence=0.8,
                        supports_media=[MediaType.MP3_128],
                        contradicts_media=[MediaType.DIGITAL_NATIVE, MediaType.HIRES_PCM],
                        description=f"Sehr wenig HF-Energie über 16 kHz (Ratio={hf_ratio:.4f}): MP3-typisch",
                    )
                )
        except Exception:
            logger.debug("ForensicsEngine: codec Analyse fehlgeschlagen", exc_info=True)
        return evidence

    def _analyze_analog_specific(self, audio, sr):
        """Analog-spezifische Analyse: Wow/Flutter und Knisterrate."""
        evidence = []
        try:
            y = (audio.flatten() if audio.ndim > 1 else audio).astype(np.float64)
            # Wow/Flutter: Pitch-Schwankung via Ableitung der Instantan-Phase
            from scipy.signal import hilbert

            analytic = hilbert(y[: min(len(y), sr * 5)])
            analytic_c = np.asarray(analytic, dtype=np.complex128)
            inst_phase = np.unwrap(np.asarray(np.angle(analytic_c), dtype=np.float64))
            inst_freq = np.diff(inst_phase) / (2.0 * np.pi / sr)
            flutter_std = float(np.std(inst_freq[:sr]))
            if flutter_std > 2.0:
                evidence.append(
                    ForensicEvidence(
                        feature="pitch_flutter_hz",
                        detected_value=flutter_std,
                        expected_range=(2.0, 20.0),
                        confidence=0.7,
                        supports_media=[MediaType.TAPE_7_5IPS, MediaType.CASSETTE_TYPE_I, MediaType.VINYL_LP_STEREO],
                        contradicts_media=[MediaType.DIGITAL_NATIVE, MediaType.CD],
                        description=f"Pitch-Schwankung ({flutter_std:.2f} Hz) typisch für analoges Medium",
                    )
                )
            # Knister-Rate (Impulse mit hoher Amplitude)
            abs_y = np.abs(y)
            threshold = float(np.percentile(abs_y, 99))
            impulse_rate = float(np.sum(abs_y > threshold * 1.5) / len(y))
            if impulse_rate > 0.001:
                evidence.append(
                    ForensicEvidence(
                        feature="crackle_rate",
                        detected_value=impulse_rate,
                        expected_range=(0.001, 0.05),
                        confidence=0.75,
                        supports_media=[MediaType.VINYL_LP_STEREO, MediaType.SHELLAC_ELECTRIC],
                        contradicts_media=[MediaType.DIGITAL_NATIVE, MediaType.CD],
                        description=f"Knister-Rate {impulse_rate:.4f}: Vinyl/Shellac-typisch",
                    )
                )
        except Exception:
            logger.debug("ForensicsEngine: analog-specific Analyse fehlgeschlagen", exc_info=True)
        return evidence

    def _generate_hypotheses(self) -> list:
        # SOTA: Evidenz-basiertes Voting
        votes: dict[MediaType, float] = {}
        for ev in self.evidence:
            for m in ev.supports_media:
                votes[m] = votes.get(m, 0) + ev.confidence
            for m in ev.contradicts_media:
                votes[m] = votes.get(m, 0) - ev.confidence
        if not votes:
            return [MediaHypothesis(media_type=MediaType.UNKNOWN, confidence=0.0, evidence=[])]
        best_media = max(votes, key=lambda k: votes[k])
        conf = min(1.0, max(0.0, votes[best_media] / (len(self.evidence) or 1)))
        return [MediaHypothesis(media_type=best_media, confidence=conf, evidence=self.evidence)]

    def _detect_transfer_chain(self, hypotheses):
        # Transfer-Kette: vollständige Medienkette vom Original zum Digitalisat
        chain: list[MediaType] = []
        if not hypotheses:
            return chain
        m = hypotheses[0].media_type

        # Basis: primäres analoges/digitales Medium
        if m == MediaType.UNKNOWN:
            return [MediaType.UNKNOWN]

        chain.append(m)

        # Prüfung auf Digitalisierungsstufen: analoge Medien haben fast immer
        # eine digitale Kopie am Ende der Kette
        analog_media = {
            MediaType.VINYL_LP_STEREO,
            MediaType.VINYL_LP_MONO,
            MediaType.VINYL_45_STEREO,
            MediaType.VINYL_45_MONO,
            MediaType.SHELLAC_ACOUSTIC,
            MediaType.SHELLAC_ELECTRIC,
            MediaType.TAPE_7_5IPS,
            MediaType.TAPE_15IPS,
            MediaType.TAPE_3_75IPS,
            MediaType.CASSETTE_TYPE_I,
            MediaType.CASSETTE_TYPE_II,
            MediaType.CASSETTE_TYPE_IV,
            MediaType.CYLINDER_EDISON,
            MediaType.CYLINDER_PATHE,
            MediaType.WIRE_RECORDING,
        }

        if m in analog_media:
            chain.append(MediaType.DIGITAL_NATIVE)

        # §Vinyl-Ghost-Detection: wenn Primärmedium cassette/tape ist aber
        # vinyl-typische Artefakte (clicks, hum, surface noise) vorhanden sind,
        # dann war die Quelle sehr wahrscheinlich Vinyl.
        # Dies erkennt den häufigen Fall: Vinyl → Cassette-Dub → Digital.
        #
        # §Robust: Prüft sowohl self.evidence (Rohdaten, unberührt von
        # _generate_hypotheses) als auch hypotheses[0].evidence.
        # self.evidence überlebt file_ext-basierte Nullierung (z.B. .mp3),
        # die hypotheses[0].evidence zurücksetzen kann.
        vinyl_ghost = False
        for _ev_source in (self.evidence, getattr(hypotheses[0], "evidence", []) if hypotheses else []):
            for ev in _ev_source if isinstance(_ev_source, list) else []:
                label = str(getattr(ev, "label", "")).lower()
                feature = str(getattr(ev, "feature", "")).lower()
                desc = str(getattr(ev, "description", "")).lower()
                combined = f"{label} {feature} {desc}"
                if any(
                    kw in combined
                    for kw in ["click", "crackle", "hum", "surface_noise", "vinyl", "riaa", "groove", "stylus"]
                ):
                    vinyl_ghost = True
                    break
            if vinyl_ghost:
                break
        if vinyl_ghost and MediaType.VINYL_LP_STEREO not in chain:
            # Füge Vinyl als Originalquelle VOR dem aktuellen Medium ein
            chain.insert(0, MediaType.VINYL_LP_STEREO)

        # MP3/Codec-Erkennung: prüfe Evidenz auf digitale Artefakte
        codec_evidence = [
            ev
            for ev in (hypotheses[0].evidence if hasattr(hypotheses[0], "evidence") else [])
            if hasattr(ev, "label")
            and any(
                kw in str(ev.label).lower()
                for kw in ["mp3", "mpeg", "codec", "aac", "compression", "lossy", "quantization"]
            )
        ]

        # Auch auf DefectType.MPEG_FRAME_LOSS / QUANTIZATION_NOISE im Scanner prüfen
        # Wenn codec-evidenz vorhanden, füge MP3_LOW als zusätzliches Glied hinzu
        if codec_evidence:
            chain.append(MediaType.MP3_128)

        # Vermeide Duplikate
        seen = set()
        unique_chain = []
        for ct in chain:
            if ct not in seen:
                seen.add(ct)
                unique_chain.append(ct)

        return unique_chain

    def _create_report(self, hypotheses, transfer_chain):
        # Mapping auf Haupttyp
        primary = (
            hypotheses[0] if hypotheses else MediaHypothesis(media_type=MediaType.UNKNOWN, confidence=0.0, evidence=[])
        )
        main_type = self._map_to_main_type(primary.media_type)
        summary = f"Erkanntes Medium: {main_type.name} (Konfidenz: {primary.confidence:.2f})\n"
        summary += f"Transfer-Chain: {[self._map_to_main_type(m).name for m in transfer_chain]}\n"
        summary += f"Mapping: {primary.media_type.name} -> {main_type.name}\n"
        for ev in self.evidence:
            summary += f"Evidenz: {ev.feature}={ev.detected_value:.2f} ({ev.description})\n"
        return ForensicReport(
            primary_media=main_type,
            primary_confidence=primary.confidence,
            alternatives=hypotheses,
            transfer_detected=len(transfer_chain) > 1,
            transfer_chain=[self._map_to_main_type(m) for m in transfer_chain],
            all_evidence=self.evidence,
            summary=summary,
        )

    def analyze(self, audio: np.ndarray, sr: int, detailed: bool = True) -> ForensicReport:
        """
        Vollständige forensische Analyse.

        Args:
            audio: Audio [samples] oder [samples, channels]
            sr: Sample Rate
            detailed: Ausführliche Analyse

        Returns:
            ForensicReport mit allen Befunden
        """
        self.evidence = []
        # Ensure mono for some analyses
        if audio.ndim == 2:
            mono = np.mean(audio, axis=1)
            stereo = audio
        else:
            mono = audio
            stereo = None
        # LAYER 1: SPEKTRALE ANALYSE
        spectral_evidence = self._analyze_spectral(mono, sr)
        self.evidence.extend(spectral_evidence)
        # LAYER 2: RAUSCH-ANALYSE
        noise_evidence = self._analyze_noise(mono, sr)
        self.evidence.extend(noise_evidence)
        # LAYER 3: ARTEFAKT-DETEKTION
        artifact_evidence = self._analyze_artifacts(mono, sr)
        self.evidence.extend(artifact_evidence)
        # LAYER 4: DYNAMIK-ANALYSE
        dynamics_evidence = self._analyze_dynamics(mono, sr)
        self.evidence.extend(dynamics_evidence)
        # LAYER 5: STEREO-ANALYSE
        if stereo is not None:
            stereo_evidence = self._analyze_stereo(stereo, sr)
            self.evidence.extend(stereo_evidence)
        # LAYER 6: CODEC-SPEZIFISCHE DETEKTION
        codec_evidence = self._analyze_codecs(mono, sr)
        self.evidence.extend(codec_evidence)
        # LAYER 7: ANALOGSPEZIFISCHE DETEKTION
        analog_evidence = self._analyze_analog_specific(mono, sr)
        self.evidence.extend(analog_evidence)
        # HYPOTHESEN GENERIEREN
        hypotheses = self._generate_hypotheses()
        # TRANSFER-DETEKTION
        transfer_chain = self._detect_transfer_chain(hypotheses)
        # REPORT GENERIEREN
        return self._create_report(hypotheses, transfer_chain)  # type: ignore[no-any-return]
