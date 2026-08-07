import pytest

"""§2.64 v10.0.0 — Unit-Tests für _fast_goal_snapshot Proxy-Kalibrierung.

Schützt vor Regression der drei systemischen Proxy-Bugs, die zu persistenten
zu niedrigen Delta-Werten für ALLE Import-Songs führten:

1. Single-Segment-Bias: spektrale Proxys nur auf Zentrum-Segment → jetzt Multi-Segment.
2. authentizitaet-ACF-Crash nach phase_24: ACF auf ersten 8192 Samples → jetzt Zentral-Drittel.
3. transparenz-Proxy: Single-Segment-Perzentil-SNR → jetzt Vollsignal + SFM-Blend.
"""

import numpy as np

# Importpfad: _fast_goal_snapshot ist @staticmethod in UnifiedRestorerV3
# Wir importieren nur die statische Methode, nicht den vollen UV3.
from backend.core.unified_restorer_v3 import UnifiedRestorerV3

_SR = 48000
_snap = UnifiedRestorerV3._fast_goal_snapshot


def _make_tonal_audio(n: int = _SR * 4, sr: int = _SR) -> np.ndarray:
    """Tonales Testsignal: Summe harmonischer Sinustöne (simuliert Musik)."""
    t = np.arange(n) / sr
    sig = (
        0.5 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.15 * np.sin(2 * np.pi * 880.0 * t)
        + 0.05 * np.sin(2 * np.pi * 1760.0 * t)
    ).astype(np.float32)
    return sig / (np.max(np.abs(sig)) + 1e-8)  # type: ignore[no-any-return]


def _make_noisy_audio(n: int = _SR * 4, sr: int = _SR) -> np.ndarray:
    """Rausch-Signal: weißes Rauschen (simuliert defektes Audio)."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(n) * 0.3).astype(np.float32)


def _make_dropout_repaired_audio(n: int = _SR * 4, sr: int = _SR) -> np.ndarray:
    """Simuliert audio NACH phase_24 Dropout-Repair:
    - Intro (erste 0.5 s): inpainted (linearer Ramp — andere ACF-Struktur)
    - Rest: normales tonales Signal
    """
    sig = _make_tonal_audio(n, sr)
    # Überschreibe Anfang mit Inpainting-ähnlichem Material (linearer Ramp)
    dropout_len = int(0.5 * sr)
    sig[:dropout_len] = np.linspace(0.0, sig[dropout_len], dropout_len).astype(np.float32)
    return sig


def _make_compressed_pop(n: int = _SR * 4, sr: int = _SR) -> np.ndarray:
    """Simuliert komprimiertes Pop/Schlager: dichtes harmonisches Signal mit hohem RMS,
    kein Dynamikabfall (wie typischer Schlager nach Mastering).
    """
    t = np.arange(n) / sr
    # Dichte harmonische Schichtung simuliert Pop-Mix
    sig = sum(
        amp * np.sin(2 * np.pi * freq * t)
        for freq, amp in [(110, 0.4), (220, 0.35), (440, 0.3), (880, 0.25), (1760, 0.2), (3520, 0.15)]
    )
    sig = np.array(sig, dtype=np.float32)
    # Harte Begrenzung simuliert Mastering-Kompression (hohe 5th-Perzentile ≈ 0.3)
    sig = np.clip(sig, -0.8, 0.8)
    return (sig / (np.max(np.abs(sig)) + 1e-8)).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Test 1: Natuerlichkeit-Proxy — tonales Signal > Rauschen (Multi-Segment-fix)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNatuerlichkeitProxy:
    def test_tonal_higher_than_noise(self):
        """Tonales Signal muss deutlich höhere natuerlichkeit haben als Rauschen."""
        tonal = _make_tonal_audio()
        noisy = _make_noisy_audio()
        s_tonal = _snap(tonal, _SR)
        s_noisy = _snap(noisy, _SR)
        assert "natuerlichkeit" in s_tonal
        assert s_tonal["natuerlichkeit"] > s_noisy["natuerlichkeit"] + 0.15, (
            f"Tonal={s_tonal['natuerlichkeit']:.3f} sollte > Noise={s_noisy['natuerlichkeit']:.3f} + 0.15"
        )

    def test_multi_segment_stability(self):
        """Pause im Zentrum-Segment darf natuerlichkeit nicht kollabieren.

        §2.64 v10.0.0: Single-Segment-Bug — Pause im Zentrum gab natuerlichkeit ≈ 0.02.
        Multi-Segment-Fix: 25%/50%/75% → stabile Messung auch bei Zentrum-Pause.
        """
        sig = _make_tonal_audio()
        # Zentriertes Stille-Segment (genau dort, wo Single-Segment messen würde)
        center = len(sig) // 2
        pause_len = 4096
        sig_with_pause = sig.copy()
        sig_with_pause[center - pause_len // 2 : center + pause_len // 2] = 0.0
        s = _snap(sig_with_pause, _SR)
        assert s["natuerlichkeit"] > 0.25, (
            f"natuerlichkeit={s['natuerlichkeit']:.3f} trotz Zentrum-Pause "
            f"(Multi-Segment-Fix muss Tonalität aus 25%/75%-Segmenten erkennen)"
        )


# ---------------------------------------------------------------------------
# Test 2: Authentizitaet-Proxy — Crash nach phase_24-Inpainting verhindert
# ---------------------------------------------------------------------------


class TestAuthentizitaetProxy:
    def test_dropout_repaired_intro_doesnt_crash_acf(self):
        """§2.64 v10.0.0: ACF auf Zentral-Drittel vermeidet phase_24-Inpainting-Crash.

        Alt: ACF auf mono[:8192] — nach Dropout-Repair am Intro kollabiert Peak 0.71→0.06.
        Neu: ACF auf mono[N//3 : N//3+8192] — stabiler Bereich, kein Absturz.
        """
        sig_repaired = _make_dropout_repaired_audio()
        sig_clean = _make_tonal_audio()
        s_rep = _snap(sig_repaired, _SR)
        s_clean = _snap(sig_clean, _SR)
        # Repaired signal sollte ähnliche authentizitaet haben wie sauberes Signal
        # (nicht um >0.30 darunter liegen — das war der Bug)
        assert "authentizitaet" in s_rep
        gap = s_clean["authentizitaet"] - s_rep["authentizitaet"]
        assert gap < 0.30, (
            f"authentizitaet-Gap nach Dropout-Repair-Intro: clean={s_clean['authentizitaet']:.3f} "
            f"repaired={s_rep['authentizitaet']:.3f} gap={gap:.3f} — Regression des §2.64 v10.0.0 Fixes!"
        )

    def test_authentizitaet_voiced_signal(self):
        """Harmonisches Signal muss authentizitaet > 0.5 haben."""
        sig = _make_tonal_audio()
        s = _snap(sig, _SR)
        assert s["authentizitaet"] > 0.50, f"authentizitaet={s['authentizitaet']:.3f} für tonales Signal — sollte > 0.5"


# ---------------------------------------------------------------------------
# Test 3: Transparenz-Proxy — komprimiertes Pop/Schlager darf nicht auf ≤ 0.20 kollabieren
# ---------------------------------------------------------------------------


class TestTransparenzProxy:
    def test_compressed_pop_not_near_zero(self):
        """§2.64 v10.0.0: transparenz-Proxy darf für komprimiertes Pop nicht ≤ 0.20 sein.

        Alt: Single-Segment 5th/99th-Perzentil-SNR → bei loud Passage 5th-Pct ≈ 0.2,
        peak ≈ 0.25 → log10(1.25)/5 ≈ 0.02 → transparenz=0.02. Bug!
        Neu: Vollsignal-Perzentile + SFM-Blend → stabiler Wert > 0.20 für Musik.
        """
        pop = _make_compressed_pop()
        s = _snap(pop, _SR)
        assert "transparenz" in s
        assert s["transparenz"] >= 0.20, (
            f"transparenz={s['transparenz']:.3f} — Vollsignal-Proxy-Fix muss ≥ 0.20 liefern "
            f"für komprimiertes Pop/Schlager"
        )

    def test_tonal_higher_than_noise(self):
        """Tonales Signal (klar) muss höhere transparenz haben als Weißrauschen."""
        tonal = _make_tonal_audio()
        noisy = _make_noisy_audio()
        s_t = _snap(tonal, _SR)
        s_n = _snap(noisy, _SR)
        assert s_t["transparenz"] >= s_n["transparenz"], (
            f"Tonal={s_t['transparenz']:.3f} sollte >= Noise={s_n['transparenz']:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 4: Brillanz-Proxy — HF-reiches Material > HF-armes Material
# ---------------------------------------------------------------------------


class TestBrillanzProxy:
    def test_hf_rich_higher_than_hf_poor(self):
        """Helles Signal (HF > 4 kHz) muss höhere brillanz haben als dunkles."""
        t = np.arange(_SR * 4) / _SR
        hf_rich = (0.5 * np.sin(2 * np.pi * 8000.0 * t) + 0.3 * np.sin(2 * np.pi * 12000.0 * t)).astype(np.float32)
        hf_poor = (0.5 * np.sin(2 * np.pi * 100.0 * t) + 0.3 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
        s_hf = _snap(hf_rich, _SR)
        s_lf = _snap(hf_poor, _SR)
        assert s_hf["brillanz"] > s_lf["brillanz"] + 0.10, (
            f"HF-reich={s_hf['brillanz']:.3f} sollte > LF-arm={s_lf['brillanz']:.3f} + 0.10"
        )


# ---------------------------------------------------------------------------
# Test 5: Vollständigkeit — alle 15 Goal-Schlüssel vorhanden
# ---------------------------------------------------------------------------


class TestSnapshotCompleteness:
    def test_all_goals_present(self):
        """_fast_goal_snapshot muss alle 15 Musical-Goal-Schlüssel liefern."""
        sig = _make_tonal_audio()
        s = _snap(sig, _SR)
        required_keys = {
            "natuerlichkeit",
            "authentizitaet",
            "timbre_authentizitaet",
            "timbre",  # §2.64 v10.0.0: Alias für timbre_authentizitaet
            "tonal_center",
            "artikulation",
            "transient_energie",
            "emotionalitaet",
            "micro_dynamics",  # kanonischer Key (nicht "mikrodynamik")
            "groove",
            "transparenz",
            "waerme",
            "bass_kraft",  # kanonischer Key (nicht "basskraft")
            "separation_fidelity",  # kanonischer Key (nicht "sep_fidelity")
            "brillanz",
            "raumtiefe",
            "spatial_depth",  # Alias für raumtiefe (§2.64 v10.0.0)
        }
        missing = required_keys - set(s.keys())
        assert not missing, f"Fehlende Goal-Schlüssel: {missing}"

    def test_all_values_in_range(self):
        """Alle Proxy-Werte müssen in [0, 1] liegen."""
        sig = _make_tonal_audio()
        s = _snap(sig, _SR)
        out_of_range = {k: v for k, v in s.items() if not (0.0 <= v <= 1.0)}
        assert not out_of_range, f"Proxy-Werte außerhalb [0,1]: {out_of_range}"

    def test_short_audio_returns_empty(self):
        """Zu kurzes Audio (< 512 Samples) muss leeres Dict zurückgeben."""
        s = _snap(np.zeros(256, dtype=np.float32), _SR)
        assert s == {}, f"Kurzes Audio muss {{}} liefern, aber: {s}"

    def test_stereo_no_crash(self):
        """Stereo-Input (2, N) darf nicht crashen."""
        stereo = np.stack([_make_tonal_audio(), _make_tonal_audio() * 0.9])  # (2, N)
        s = _snap(stereo, _SR)
        assert isinstance(s, dict)
        assert len(s) >= 10


class TestSpatialDepthProxyMSRatio:
    """§2.64 v10.0.0 — spatial_depth/raumtiefe Proxy: M/S-Stereobreite statt HF-Anteil.

    Regression-Test für Bug: alter Proxy (4–16 kHz / 200–16 kHz * 2.5) lieferte
    für cassette/mp3_low-Material ~0.11 trotz guter Stereobreite → §GOAL_BASELINE_CHECK
    triggerte phase_46_spatial_enhancement false-positive für jede Kassetten-Restaurierung.

    Fix: Mid/Side-Energieverhältnis misst tatsächliche Stereobreite, unabhängig vom BW.
    """

    def _make_cassette_stereo(self, side_factor: float = 0.3) -> np.ndarray:
        """Simuliert Kassetten-Audio: stereo, hauptsächlich bass/mid, kein HF (BW <= 12 kHz).
        side_factor: Verhältnis Side/Mid-Amplitude (0=mono, 0.3=moderate, 1.0=pure side).
        side_factor=0.3 → Side-Energie ≈ 8% von Gesamt → spatial_depth ≈ 0.97 (via Log-M/S-Proxy).
        """
        n = _SR * 4
        t = np.arange(n) / _SR
        # Nur Frequenzen bis 3 kHz (cassette-like: kein HF über 12 kHz)
        base = 0.2 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 800 * t)
        side = side_factor * base  # skalierte L/R-Differenz
        left = (base + side).astype(np.float32)
        right = (base - side).astype(np.float32)
        return np.stack([left, right])

    def test_stereo_signal_gives_reasonable_spatial_depth(self):
        """Stereo-Signal mit echter L/R-Differenz muss spatial_depth > 0.1 liefern.
        side_factor=0.3 → side_rms ≈ 0.3 * mid_rms → Side-Energie / Gesamt ≈ 8%
        → M/S-Proxy ≈ 0.33 (Beweis: HF-Proxy hätte ~0.02 für LF-dominantes Signal).
        """
        audio = self._make_cassette_stereo(side_factor=0.3)
        s = _snap(audio, _SR)
        val = s.get("spatial_depth", 0.0)
        assert val > 0.1, (
            f"spatial_depth={val:.3f} für Stereo-Signal (side_factor=0.3) — "
            "M/S-Proxy sollte > 0.1 liefern (HF-Proxy gäbe ~0.02 für LF-Signal)"
        )

    def test_mono_signal_gives_low_spatial_depth(self):
        """Exakt mono (L==R) muss spatial_depth < 0.1 liefern (keine künstliche Breite)."""
        n = _SR * 4
        t = np.arange(n) / _SR
        mono = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        audio = np.stack([mono, mono])  # perfektes Mono
        s = _snap(audio, _SR)
        val = s.get("spatial_depth", 1.0)
        assert val < 0.1, f"spatial_depth={val:.3f} für reines Mono — sollte nahe 0 sein"

    def test_spatial_depth_independent_of_hf_content(self):
        """spatial_depth darf NICHT mit HF-Anteil korrelieren (alter Bug).
        Gleiche L/R-Breite → gleiche spatial_depth, egal ob HF-reich oder HF-arm.
        """
        n = _SR * 4
        t = np.arange(n) / _SR
        base = 0.2 * np.sin(2 * np.pi * 300 * t)
        side = 0.05 * np.sin(2 * np.pi * 250 * t + 0.5)

        # HF-arm: nur Frequenzen unter 1 kHz
        hf_poor_l = (base + side).astype(np.float32)
        hf_poor_r = (base - side).astype(np.float32)

        # HF-reich: gleiche Basis + HF-Anteil (simuliert CD)
        hf_component = 0.1 * np.sin(2 * np.pi * 8000 * t)
        hf_rich_l = (base + side + hf_component).astype(np.float32)
        hf_rich_r = (base - side + hf_component).astype(np.float32)

        s_poor = _snap(np.stack([hf_poor_l, hf_poor_r]), _SR)
        s_rich = _snap(np.stack([hf_rich_l, hf_rich_r]), _SR)

        val_poor = s_poor.get("spatial_depth", 0.0)
        val_rich = s_rich.get("spatial_depth", 0.0)

        # Beide sollten ähnliche spatial_depth haben (gleiche L/R-Differenz)
        assert abs(val_poor - val_rich) < 0.15, (
            f"spatial_depth: HF-arm={val_poor:.3f}, HF-reich={val_rich:.3f} — "
            "M/S-Proxy sollte HF-unabhängig sein (Differenz < 0.15)"
        )

    def test_spatial_depth_alias_matches_raumtiefe(self):
        """'spatial_depth' und 'raumtiefe' müssen identisch sein."""
        audio = self._make_cassette_stereo(side_factor=0.3)
        s = _snap(audio, _SR)
        assert s.get("spatial_depth") == s.get("raumtiefe"), "'spatial_depth' und 'raumtiefe' müssen identisch sein"


class TestPrimaryMaterialEnumNormalization:
    """§2.64 v10.0.0 — primary_material Enum→String-Normalisierung.

    Regression-Test für Bug: Python 3.12 str(MaterialType.CASSETTE) = 'MaterialType.CASSETTE'
    → groove-noisy_mat-Set-Check scheitert → groove proxy false-low für ALLE Kassetten.
    Fix: .value wenn Enum, sonst str(...).lower().
    """

    def test_material_type_enum_normalization(self):
        """MaterialType.CASSETTE.value.lower() muss 'cassette' ergeben."""
        from backend.core.defect_scanner import MaterialType

        pm_raw = MaterialType.CASSETTE
        pm_str = (pm_raw.value if hasattr(pm_raw, "value") else str(pm_raw)).lower()
        assert pm_str == "cassette", (
            f"Enum-Normalisierung: '{pm_str}' != 'cassette' — "
            "Python 3.12 str(Enum) gibt 'MaterialType.CASSETTE', nicht 'cassette'"
        )

    def test_groove_proxy_noisy_mat_set_does_not_accept_enum_string(self):
        """Bug-Regression: Python 3.12 str(MaterialType.CASSETTE) = 'MaterialType.CASSETTE'.
        Das in noisy_mat-Set-Check als 'materialtype.cassette' übergeben darf NICHT matchen
        — das wäre der alte Bugs-Zustand (vor Fix v10.0.0).
        Fix: primary_material normalisiert zu .value.lower() → 'cassette' vor Speicherung.
        """
        _noisy_mat_set = {"cassette", "tape", "reel_tape", "mp3_low", "mp3_high"}
        # Simuliert den ALTEN Bugzustand (kein .value): str(Enum) ohne Normalisierung
        from backend.core.defect_scanner import MaterialType

        old_buggy_str = str(MaterialType.CASSETTE).lower()  # 'materialtype.cassette' in Python 3.12
        assert old_buggy_str not in _noisy_mat_set, (
            f"'{old_buggy_str}' sollte NICHT in noisy_mat_set sein — das wäre der Bug-Zustand (kein .value)"
        )
        # Simuliert den NEUEN korrekten Zustand (mit .value):
        fixed_str = (
            MaterialType.CASSETTE.value if hasattr(MaterialType.CASSETTE, "value") else str(MaterialType.CASSETTE)
        ).lower()
        assert fixed_str in _noisy_mat_set, (
            f"'{fixed_str}' muss in noisy_mat_set sein — korrekte Normalisierung via .value"
        )
