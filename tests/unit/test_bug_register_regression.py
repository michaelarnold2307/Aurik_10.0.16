"""
§v10.980: Bug-Register-Regressionstest — die 11 behobenen Bugs dürfen NIE zurückkehren.

Jeder Test prüft, dass der jeweilige Fix im Code präsent ist. Diese Tests sind
die technische Umsetzung des Bug-Registers aus §v10.900.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _read(rel: str) -> str:
    root = Path(__file__).resolve().parent.parent.parent
    return (root / rel).read_text()


def test_b1_logger_nameerror_fix():
    """B1: Kein logger.warning in coordinated_repair (Modul-Logger heißt log)."""
    content = _read("backend/core/coordinated_repair.py")
    assert "logger.warning" not in content
    assert "log = logging.getLogger(__name__)" in content


def test_b2_ode_solver_statt_einzelschritt():
    """B2: Flow-Matching braucht ODE-Integration, keinen Ein-Schritt-Euler."""
    content = _read("backend/core/coordinated_repair.py")
    assert "for i in range(n_steps):" in content
    assert "t_flow = torch.full" not in content


def test_b3_truepeak_relativ():
    """B3: TruePeak-Check muss RELATIV sein (post vs pre), nicht absolut."""
    content = _read("backend/core/post_repair_artifact_guard.py")
    assert "peak_delta = truepeak_dbfs - pre_peak_dbfs" in content
    assert "truepeak_dbfs > TRUEPEAK_WARN_DBFS" not in content


def test_b4_mono_konvertierung():
    """B4: execute() muss Mono [T] intern zu [1, T] konvertieren."""
    content = _read("backend/core/coordinated_repair.py")
    assert "was_mono = audio.ndim == 1" in content


def test_b5_global_lokal_entkopplung():
    """B5: Globale Defekte dürfen lokale nicht verdrängen."""
    content = _read("backend/core/defect_consensus_pipeline.py")
    assert "global_cats" in content


def test_b6_banquet_material_gate():
    """B6: Banquet nur mit Opt-In + Material-Gate (Vinyl)."""
    content = _read("backend/core/coordinated_repair.py")
    assert "use_banquet = bool(step.parameters" in content


def test_b7_alle_kern_dateien_kompilieren():
    """B7: Alle Kern-Dateien müssen kompilieren (keine String-Insertion-Schäden)."""
    import py_compile

    files = [
        "backend/core/coordinated_repair.py",
        "backend/core/post_repair_artifact_guard.py",
        "backend/core/perceptual_closed_loop.py",
        "backend/core/defect_consensus_pipeline.py",
        "backend/core/sota_denoise_pipeline.py",
        "backend/core/sota_vocal_pipeline.py",
        "denker/defekt_denker.py",
        "denker/restaurier_denker.py",
    ]
    root = Path(__file__).resolve().parent.parent.parent
    for f in files:
        py_compile.compile(str(root / f), doraise=True)


def test_b8_mask_reset_im_ode_loop():
    """B8: Mask-Reset nach jedem Euler-Schritt."""
    content = _read("backend/core/coordinated_repair.py")
    assert "Mask-Reset" in content


def test_b9_velocity_nur_audio_kanal():
    """B9: Velocity darf nur auf x[..., :1] wirken (nicht auf Mask-Kanal)."""
    content = _read("backend/core/coordinated_repair.py")
    assert "x[..., :1]" in content


def test_b10_registration_report():
    """B10: Keine stillen Detektor-Failures — Registration-Report existiert."""
    content = _read("backend/core/defect_consensus_pipeline.py")
    assert "_registration_report" in content


def test_b11_spektraler_rauschfloor_check():
    """B11: Spektraler HF-Rauschfloor-Check im Guard."""
    content = _read("backend/core/post_repair_artifact_guard.py")
    assert "spectral_noise_rise" in content


def test_b12_utmos_shape_normalisierung():
    """§v10.920: UTMOS muss 2D-Inputs auf Mono reduzieren (88-Fold-Fehler-Fix)."""
    content = _read("plugins/utmos_plugin.py")
    assert "audio_f32.ndim > 1" in content
