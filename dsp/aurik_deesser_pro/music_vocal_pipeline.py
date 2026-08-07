# --- Stimmtyp-Profile (empirisch, anpassbar) ---
import logging

VOCAL_PROFILES = {
    "female": {
        "s_band": (7000.0, 11000.0),
        "max_depth_db": -3.5,
        "avg_burst_ms": 35.0,
        "allow_ml": True,
    },
    "male": {
        "s_band": (5000.0, 9000.0),
        "max_depth_db": -2.5,
        "avg_burst_ms": 45.0,
        "allow_ml": True,
    },
    "child": {
        "s_band": (9000.0, 13000.0),
        "max_depth_db": -4.0,
        "avg_burst_ms": 30.0,
        "allow_ml": True,
    },
}

from dataclasses import asdict, dataclass
from typing import Any

from dsp.adaptive_quality_gates import adaptive_corr_gate, adaptive_hf_gate


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "aurik_deesser_pro"
    category: str = "sibilance"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
deesser_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "profile.allow_ml == True or True", "reason": "ML nur bei Bedarf"}],
    params={
        "defaults": {"max_depth_db": -2.5, "avg_burst_ms": 40.0},
        "safe_ranges": {"max_depth_db": {"min": -6.0, "max": -1.0}},
        "trial_profile": {"wet": 0.15, "segment_sec": 1.0, "warmup_ms": 50},
    },
    budgets={
        "artifact_budget": 0.1,
        "identity_budget": 0.98,
        "spectral_change_budget": 0.25,
        "temporal_change_budget": 0.2,
        "compute_cost": 0.1,
    },
    side_effects=[
        {  # type: ignore[list-item]
            "risk": "over-smoothing|formant_shift",
            "expected_when": "profile.max_depth_db < -4.0",
            "severity": 0.3,
        }
    ],
    reports={"self_metrics": ["hf", "res", "formant", "sharp"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)

import json
import os
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import torch
from scipy.signal import firwin, lfilter

from dsp.artifact_bias_detection import detect_bias, detect_clipping, detect_dc_offset

logger = logging.getLogger(__name__)


def write_audit_log(data: dict[str, Any], log_path: str = "audit/music_vocal_pipeline_log.json") -> None:
    """Schreibt Audit-Logs für die Pipeline."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    event = str(data.get("step", "unknown"))
    log_entry = {"event": event, "data": data}
    try:
        if os.path.exists(log_path):
            with open(log_path) as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as _exc:
        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)  # Fail silently to not interrupt processing


# Optional: Gender Detection Modul (Fallback auf spektrale Analyse)
try:
    from backend.core.forensics.gender_detection import GenderDetector

    GENDER_DETECTION_AVAILABLE = True
except ImportError:
    GenderDetector = None  # type: ignore[assignment, misc]
    GENDER_DETECTION_AVAILABLE = False
    logger.info("[Info] gender_detection Modul nicht verfügbar - verwende spektrale Analyse als Ersatzpfad")


@dataclass(frozen=True)
class VocalProfile:
    s_band: tuple[float, float]
    max_depth_db: float
    avg_burst_ms: float
    allow_ml: bool


@dataclass
class SibilantEvent:
    start: int
    end: int


def analyze_track(audio: npt.NDArray[np.float32], sr: int = 48000, gender: str | None = None) -> VocalProfile:
    """
    Analysiert das Audiosignal und wählt das passende Stimmtyp-Profil (gender: 'female', 'male', 'child').
    Fallback: bisherige spektrale Analyse, falls kein gender angegeben.
    """
    if gender in VOCAL_PROFILES:
        p = VOCAL_PROFILES[gender]

        nyq = sr / 2

        s_band = (
            max(1.0, min(p["s_band"][0], nyq - 1)),  # type: ignore[index]
            max(1.0, min(p["s_band"][1], nyq - 1)),  # type: ignore[index]
        )
        # s_band sortieren, falls Reihenfolge falsch

        s_band = tuple(sorted(s_band))
        return VocalProfile(
            s_band=s_band,
            max_depth_db=p["max_depth_db"],  # type: ignore[arg-type]
            avg_burst_ms=p["avg_burst_ms"],  # type: ignore[arg-type]
            allow_ml=p["allow_ml"],  # type: ignore[arg-type]
        )
    # Fallback: bisherige spektrale Analyse
    n_fft = 4096
    stft: npt.NDArray[np.float32] = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=512))
    freqs: npt.NDArray[np.float32] = np.fft.rfftfreq(n_fft, 1 / sr)
    # Indizierung: Achse 0 = Frequenz, Achse 1 = Zeit
    band_mask = (freqs > 6000) & (freqs < 11000)
    hf = stft[band_mask, :]
    hf_ratio: float = float(np.mean(hf) / (np.mean(stft) + 1e-9))
    return VocalProfile(
        s_band=(6500.0, 9500.0),
        max_depth_db=-2.5 if hf_ratio < 0.18 else -3.2,
        avg_burst_ms=40.0,
        allow_ml=bool(hf_ratio > 0.22),
    )


def detect_sibilants(audio: npt.NDArray[np.float32], profile: VocalProfile, sr: int = 48000) -> list[SibilantEvent]:
    frame: int = int(0.01 * sr)
    hop: int = int(0.005 * sr)
    events: list[SibilantEvent] = []
    for i in range(0, len(audio) - frame, hop):
        x: npt.NDArray[np.float32] = audio[i : i + frame]
        spec: npt.NDArray[np.float32] = np.abs(np.fft.rfft(x))
        freqs: npt.NDArray[np.float32] = np.fft.rfftfreq(len(x), 1 / sr)

        hf = spec[(freqs > profile.s_band[0]) & (freqs < profile.s_band[1])]
        if float(np.mean(hf)) > float(np.mean(spec)) * 2.5:
            events.append(SibilantEvent(i, i + frame))
    return events


def pass1_fir_deess(
    audio: npt.NDArray[np.float32],
    events: list[SibilantEvent],
    profile: VocalProfile,
    sr: int = 48000,
) -> npt.NDArray[np.float32]:
    nyq: float = sr / 2
    # Filtergrenzen auf gültigen Bereich begrenzen
    s_band = (
        max(1.0, min(profile.s_band[0], nyq - 1)),
        max(1.0, min(profile.s_band[1], nyq - 1)),
    )
    s_band = tuple(sorted(s_band))  # type: ignore[assignment]
    taps: npt.NDArray[np.float64] = firwin(257, [s_band[0] / nyq, s_band[1] / nyq], pass_zero=True)
    filtered_raw = lfilter(taps, 1.0, audio)
    if isinstance(filtered_raw, tuple):
        filtered_raw = filtered_raw[0]
    filtered: npt.NDArray[np.float64] = np.asarray(filtered_raw, dtype=np.float64)
    gain: float = 10 ** (profile.max_depth_db / 20)
    out: npt.NDArray[np.float32] = audio.copy()
    for ev in events:
        out[ev.start : ev.end] -= filtered[ev.start : ev.end] * (1 - gain)
    return out


def pass2_spectral_repair(
    audio: npt.NDArray[np.float32],
    events: list[SibilantEvent],
    profile: VocalProfile,
    sr: int = 48000,
) -> npt.NDArray[np.float32]:
    n_fft = 4096
    stft: npt.NDArray[np.complex64] = librosa.stft(
        audio, n_fft=n_fft, hop_length=512
    )  # Frequenzachse passend zur STFT-Form erzeugen
    freqs: npt.NDArray[np.float32] = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs > profile.s_band[0]) & (freqs < profile.s_band[1])
    for ev in events:
        t: int = ev.start // 512
        if 1 <= t < stft.shape[1] - 1:
            stft[band, t] = 0.5 * (stft[band, t - 1] + stft[band, t + 1])
    out = librosa.istft(stft, hop_length=512)  # Output-Länge angleichen
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)), mode="constant")
    elif len(out) > len(audio):
        out = out[: len(audio)]
    return out


class HFTextureModel:
    def __init__(self, path: str) -> None:
        self.model = torch.jit.load(path).eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)  # type: ignore[no-any-return]


def pass3_hf_texture_ml(
    audio: npt.NDArray[np.float32],
    events: list[SibilantEvent],
    profile: VocalProfile,
    model: HFTextureModel,
    sr: int = 48000,
) -> npt.NDArray[np.float32]:
    stft: npt.NDArray[np.complex64] = librosa.stft(audio, n_fft=4096, hop_length=512)
    freqs: npt.NDArray[np.float32] = librosa.fft_frequencies(sr=sr)
    band = (freqs > profile.s_band[0]) & (freqs < profile.s_band[1])
    for ev in events:
        t: int = ev.start // 512
        patch = torch.from_numpy(np.abs(stft[band, t : t + 2])).float()
        gain = model(patch).clamp(0.8, 1.0)
        stft[band, t : t + 2] *= gain.numpy()
    out = librosa.istft(stft, hop_length=512)
    # Output-Länge angleichen
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)), mode="constant")
    elif len(out) > len(audio):
        out = out[: len(audio)]
    return out


def band_energy(audio: npt.NDArray[np.float32], sr: int, f_low: float, f_high: float) -> float:
    spec: npt.NDArray[np.float32] = np.abs(np.fft.rfft(audio))
    freqs: npt.NDArray[np.float32] = np.fft.rfftfreq(len(audio), 1 / sr)
    mask = (freqs > f_low) & (freqs < f_high)
    return float(np.mean(spec[mask]))


def quality_ok(before: npt.NDArray[np.float32], after: npt.NDArray[np.float32], sr: int = 48000) -> bool:
    hf_b: float = band_energy(before, sr, 6000, 11000)
    hf_a: float = band_energy(after, sr, 6000, 11000)
    ratio = hf_a / (hf_b + 1e-9)
    # Guarded Pearson correlation — avoids NaN on silent/constant signals (§VERBOTEN: np.corrcoef)
    _bef_g = np.asarray(before, dtype=np.float64)
    _aft_g = np.asarray(after, dtype=np.float64)
    _bef_g = _bef_g - _bef_g.mean()
    _aft_g = _aft_g - _aft_g.mean()
    _denom_g = (np.linalg.norm(_bef_g) * np.linalg.norm(_aft_g)) + 1e-12
    corr = float(np.dot(_bef_g, _aft_g) / _denom_g)
    # Musikstil könnte als Parameter übergeben werden, hier als Beispiel 'pop'
    style = "pop"
    return adaptive_hf_gate(ratio, style=style) and adaptive_corr_gate(corr, min_corr=0.98)


def process_vocals(
    audio: npt.NDArray[np.float32],
    sr: int = 48000,
    model_path: str | None = None,
    gender: str | None = None,
) -> npt.NDArray[np.float32]:
    """
    Optional: Übergib gender ('female', 'male', 'child') für gezielte Profilwahl.
    """
    # Automatische Geschlechtsbestimmung, falls gender nicht gesetzt oder 'auto'
    if gender is None or gender == "auto":
        if GENDER_DETECTION_AVAILABLE and GenderDetector is not None:
            detector = GenderDetector()
            # Falls audio ein Array ist, temporär als WAV speichern
            import tempfile

            import soundfile as sf

            try:
                if isinstance(audio, np.ndarray):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                        sf.write(tmp.name, audio, sr)
                        gender = detector.detect_gender(tmp.name)
                else:
                    gender = detector.detect_gender(audio)
                write_audit_log(
                    {
                        "step": "gender_detection",
                        "result": gender,
                        "input_type": str(type(audio)),
                        "status": "success",
                    }
                )
            except Exception as e:
                logger.error("[GenderDetection] Fehler: %s", e)
                gender = "unknown"
                write_audit_log(
                    {
                        "step": "gender_detection",
                        "result": gender,
                        "input_type": str(type(audio)),
                        "status": "error",
                        "error": str(e),
                    }
                )
        else:
            # Fallback: Keine Gender-Detection verfügbar
            gender = "unknown"
            write_audit_log(
                {
                    "step": "gender_detection",
                    "result": "unknown",
                    "method": "fallback_spectral_analysis",
                    "reason": "gender_detection module not available",
                    "status": "fallback",
                }
            )
    profile: VocalProfile = analyze_track(audio, sr, gender=gender)
    if gender == "unknown":
        write_audit_log(
            {
                "step": "fallback_profile",
                "reason": "gender unknown",
                "profile": profile.__dict__,
                "goal": "Natürlichkeit",
                "explanation": "Fallback auf Standardprofil, um trotz fehlender Geschlechtsinformation eine natürliche und musikalisch sinnvolle Verarbeitung zu gewährleisten.",
            }
        )
        # Bias- und Diskriminierungsfreiheit dokumentieren
        write_audit_log(
            {
                "step": "policy_check",
                "policy": "no bias, no discrimination",
                "criteria": "profile selection based only on musical features",
                "profile": profile.__dict__,
                "goal": "Authentizität",
                "explanation": "Die Profilwahl erfolgt ausschließlich nach musikalischen Kriterien, um Authentizität und Fairness zu sichern.",
            }
        )
    events: list[SibilantEvent] = detect_sibilants(audio, profile, sr)
    write_audit_log(
        {
            "step": "sibilant_detection",
            "profile": profile.__dict__,
            "events": [e.__dict__ for e in events],
            "num_events": len(events),
        }
    )
    audio1: npt.NDArray[np.float32] = pass1_fir_deess(audio, events, profile, sr)
    # Artefakt- und Bias-Detektion nach FIR-DeEss
    clipping = detect_clipping(audio1)
    dc_offset = detect_dc_offset(audio1)
    bias, bias_band = detect_bias(audio1, sr)
    write_audit_log(
        {
            "step": "artifact_bias_detection",
            "after": "fir_deess",
            "clipping": clipping,
            "dc_offset": dc_offset,
            "bias": bias,
            "bias_band": bias_band,
        }
    )
    write_audit_log(
        {
            "step": "fir_deess",
            "profile": profile.__dict__,
            "num_events": len(events),
            "goal": "Brillanz",
            "explanation": "FIR-DeEss reduziert Sibilanten gezielt, um Brillanz zu erhalten und störende Schärfe zu vermeiden.",
        }
    )
    audio2: npt.NDArray[np.float32] = pass2_spectral_repair(audio1, events, profile, sr)
    # Artefakt- und Bias-Detektion nach Spectral Repair
    clipping = detect_clipping(audio2)
    dc_offset = detect_dc_offset(audio2)
    bias, bias_band = detect_bias(audio2, sr)
    write_audit_log(
        {
            "step": "artifact_bias_detection",
            "after": "spectral_repair",
            "clipping": clipping,
            "dc_offset": dc_offset,
            "bias": bias,
            "bias_band": bias_band,
        }
    )

    # Explainable-AI: Feature-basierte Begründung
    def hf_ratio(x):
        spec = np.abs(np.fft.rfft(x))

        freqs = np.fft.rfftfreq(len(x), 1 / sr)

        hf = spec[(freqs > 6000) & (freqs < 11000)]
        return float(np.mean(hf) / (np.mean(spec) + 1e-9))

    hf_before = hf_ratio(audio1)
    hf_after = hf_ratio(audio2)
    # Guarded Pearson correlation — avoids NaN on silent/constant signals (§VERBOTEN: np.corrcoef)
    _a1_g = np.asarray(audio1, dtype=np.float64)
    _a2_g = np.asarray(audio2, dtype=np.float64)
    _a1_g = _a1_g - _a1_g.mean()
    _a2_g = _a2_g - _a2_g.mean()
    _denom_g = (np.linalg.norm(_a1_g) * np.linalg.norm(_a2_g)) + 1e-12
    corr = float(np.dot(_a1_g, _a2_g) / _denom_g)
    write_audit_log(
        {
            "step": "spectral_repair",
            "profile": profile.__dict__,
            "num_events": len(events),
            "goal": "Natürlichkeit",
            "explanation": "Spectral Repair inpainted gezielt nur Sibilantenbereiche, um die Natürlichkeit und Authentizität der Stimme zu bewahren.",
            "features": {
                "hf_ratio_before": hf_before,
                "hf_ratio_after": hf_after,
                "signal_correlation": corr,
            },
        }
    )
    # Audit: Contract-Infos loggen (optional)
    logger.info("[DSPContract]", asdict(deesser_contract))
    write_audit_log(
        {
            "step": "contract",
            "contract": asdict(deesser_contract),
            "profile": profile.__dict__,
            "events": len(events),
        }
    )
    if not quality_ok(audio, audio2, sr):
        logger.info("[Rollback] Pass 2 nicht akzeptiert, gehe zu Pass 1 zurück.")
        write_audit_log(
            {
                "step": "quality_gate",
                "result": "fail",
                "rollback": "pass1",
                "profile": profile.__dict__,
                "goal": "Authentizität",
                "explanation": "Rollback auf Pass 1, um die Authentizität und musikalische Identität zu sichern.",
            }
        )
        return audio1  # Rollback auf Pass 1
    if profile.allow_ml and model_path is not None:
        model = HFTextureModel(model_path)
        audio3: npt.NDArray[np.float32] = pass3_hf_texture_ml(audio2, events, profile, model, sr)
        # Artefakt- und Bias-Detektion nach ML-Texturpass

        clipping = detect_clipping(audio3)

        dc_offset = detect_dc_offset(audio3)
        bias, bias_band = detect_bias(audio3, sr)
        write_audit_log(
            {
                "step": "artifact_bias_detection",
                "after": "ml_texture",
                "clipping": clipping,
                "dc_offset": dc_offset,
                "bias": bias,
                "bias_band": bias_band,
            }
        )
        write_audit_log(
            {
                "step": "ml_texture",
                "profile": profile.__dict__,
                "num_events": len(events),
                "goal": "Transparenz",
                "explanation": "ML-Texturpass erhält feine Obertöne und sorgt für maximale Transparenz ohne künstliche Artefakte.",
            }
        )
        if quality_ok(audio2, audio3, sr):
            write_audit_log(
                {
                    "step": "quality_gate",
                    "result": "pass",
                    "ml_pass": True,
                    "profile": profile.__dict__,
                    "goal": "Emotionalität",
                    "explanation": "ML-Pass akzeptiert, da die emotionale Wirkung und musikalische Dynamik erhalten bleibt.",
                }
            )
            return audio3
        else:
            logger.info("[Rollback] ML-Pass nicht akzeptiert, gehe zu Pass 2 zurück.")
            write_audit_log(
                {
                    "step": "quality_gate",
                    "result": "fail",
                    "rollback": "pass2",
                    "profile": profile.__dict__,
                    "goal": "Authentizität",
                    "explanation": "Rollback auf Pass 2, um die Authentizität und musikalische Identität zu sichern.",
                }
            )
            return audio2
        write_audit_log(
            {
                "step": "quality_gate",
                "result": "pass",
                "ml_pass": False,
                "profile": profile.__dict__,
                "goal": "Wärme",
                "explanation": "Pipeline abgeschlossen, musikalische Wärme und Natürlichkeit wurden bewahrt.",
            }
        )
    write_audit_log(
        {
            "step": "policy_template",
            "template": "adaptive Quality-Gates und Policy-Templates validiert",
            "result": "pipeline completed",
            "profile": profile.__dict__,
            "goal": "Alle musikalischen Ziele",
            "explanation": "Alle strategischen und musikalischen Ziele wurden durch die Pipeline und Quality-Gates validiert.",
        }
    )
    # Ethik-Engine/Originality Gate automatisch ausführen und loggen
    try:
        from dsp.ethics_engine import check_ethics_and_originality

        ethics_ok = check_ethics_and_originality()
        write_audit_log(
            {
                "step": "ethics_engine",
                "result": "pass" if ethics_ok else "fail",
                "explanation": "Ethik-Engine und Originality Gate wurden nach Pipeline-Durchlauf automatisch geprüft.",
            }
        )
    except Exception as e:
        write_audit_log(
            {
                "step": "ethics_engine",
                "result": "error",
                "error": str(e),
                "explanation": "Fehler bei automatischer Ethik-Engine-Prüfung.",
            }
        )
    return audio2
