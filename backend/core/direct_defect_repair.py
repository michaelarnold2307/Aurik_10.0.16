"""DirectDefectRepair — Standalone, aggressive repair for tape head & vocal scratches.

Runs independently of the Aurik pipeline. Can be called on any WAV/FLAC file.
Optimized for: cassette dubs of vinyl records with tape head damage.

Usage:
    python -c "
    from backend.core.direct_defect_repair import repair_file
    repair_file('input.wav', 'output.wav')
    "
"""

from __future__ import annotations

import logging

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def repair_file(input_path: str, output_path: str | None = None) -> dict:
    """Load, repair, and save. Returns report dict."""
    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]  # Keep only first 2 channels

    repair = DirectDefectRepair()
    result, report = repair.repair(audio, sr)

    if output_path is None:
        output_path = input_path.replace(".wav", "_repaired.wav").replace(".flac", "_repaired.flac")

    sf.write(output_path, result.T if result.ndim == 2 else result, sr)
    logger.info("gespeichert: %s", output_path)
    return report


class DirectDefectRepair:
    """Aggressive standalone repair — no pipeline dependencies."""

    def repair(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        result = np.asarray(audio, dtype=np.float32).copy()
        mono = np.mean(result, axis=0) if result.ndim == 2 else result
        report = {}

        # ── 1. PRECISION DROPOUT REPAIR (MAD-basiert, chirurgisch) ──
        from backend.core.precision_dropout_repair import repair_dropouts_precise

        result, found, repaired = repair_dropouts_precise(result, sr)
        report["dropouts_found"] = found
        report["dropouts_repaired"] = repaired

        # ── 2. VOCAL SCRATCHES (very aggressive) ──
        scratches = self._find_scratches_aggressive(mono, sr)
        report["scratches_found"] = len(scratches)
        if scratches:
            result = self._repair_scratches(result, sr, scratches)
            report["scratches_repaired"] = len(scratches)

        # ── 3. AZIMUTH ──
        if result.ndim == 2 and result.shape[0] >= 2:
            deg = self._measure_azimuth(result, sr)
            report["azimuth_deg"] = round(deg, 2)
            if abs(deg) > 3:
                result = self._fix_azimuth(result, sr, deg)
                report["azimuth_fixed"] = True

        return np.clip(result, -1.0, 1.0).astype(np.float32), report

    def _find_dropouts_aggressive(self, audio: np.ndarray, sr: int):
        """Multi-scale dropout detection: 2ms, 5ms, 10ms windows."""
        dropouts = []
        for win_ms in [2.0, 5.0, 10.0, 20.0]:
            win = int(sr * win_ms / 1000)
            if win < 4:
                continue
            hop = max(1, win // 4)
            for i in range(win, len(audio) - win, hop):
                pre = float(np.sqrt(np.mean(audio[i - win : i] ** 2) + 1e-12))
                post = float(np.sqrt(np.mean(audio[i : i + win] ** 2) + 1e-12))
                if pre > 1e-8 and post > 1e-8:
                    db = 20 * np.log10(post / pre)
                    if -15 < db < -2:  # Very aggressive: -2dB to -15dB
                        dropouts.append((i, min(len(audio), i + win), db))
        return self._merge_regions(dropouts, 10, sr)

    def _find_scratches_aggressive(self, audio: np.ndarray, sr: int):
        """Finds sharp transients in 300-4000Hz vocal range."""
        scratches = []
        win = int(sr * 0.006)  # 6ms
        hop = max(1, win // 4)
        for i in range(win, len(audio) - win, hop):
            seg = audio[i - win : i + win]
            fft = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
            vb = (freqs >= 300) & (freqs <= 4000)
            vb_e = float(np.sum(fft[vb])) if np.any(vb) else 0
            total_e = float(np.sum(fft)) + 1e-12
            if vb_e / total_e < 0.25:
                continue

            local_rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
            peak = float(np.max(np.abs(audio[max(0, i - 3) : min(len(audio), i + 3)])))
            if peak > local_rms * 2.2:  # ~7dB — very aggressive
                s0 = i
                while s0 > 0 and abs(audio[s0]) > abs(audio[i]) * 0.25:
                    s0 -= 1
                s1 = i
                while s1 < len(audio) - 1 and abs(audio[s1]) > abs(audio[i]) * 0.25:
                    s1 += 1
                if 2 <= (s1 - s0) <= int(sr * 0.004):  # max 4ms
                    scratches.append((s0, s1, 20 * np.log10(peak / local_rms)))
        return self._merge_regions(scratches, 3, sr)

    def _repair_dropouts(self, audio, sr, dropouts):
        """Cubic interpolation repair for each channel."""
        result = np.asarray(audio, dtype=np.float32).copy()
        n = result.shape[-1]
        for ch in range(result.shape[0]):
            ch_data = result[ch]
            for s0, s1, db in dropouts:
                if s1 <= s0:
                    continue
                try:
                    pts = min(8, max(2, (s1 - s0) // 2))
                    x = np.linspace(s0 - pts, s1 + pts, 5, dtype=np.float64)
                    y = np.array([ch_data[min(n - 1, max(0, int(xi)))] for xi in x], dtype=np.float64)
                    coeffs = np.polyfit(x, y, min(3, len(x) - 1))
                    xi = np.arange(s0, s1, dtype=np.float64)
                    yi = np.polyval(coeffs, xi)
                    result[ch, s0:s1] = yi.astype(np.float32)
                except Exception as _e:
                    logger.debug("direct_defect_repair: unkritisch exception: %s", _e)
        return result

    def _repair_scratches(self, audio, sr, scratches):
        """Linear interpolation for scratches."""
        result = np.asarray(audio, dtype=np.float32).copy()
        n = result.shape[-1]
        for ch in range(result.shape[0]):
            ch_data = result[ch]
            for s0, s1, db in scratches:
                if s1 <= s0:
                    continue
                pre = ch_data[max(0, s0 - 1)]
                post = ch_data[min(n - 1, s1 + 1)]
                interp = np.linspace(pre, post, s1 - s0 + 2, dtype=np.float32)[1:-1]
                result[ch, s0:s1] = interp
        return result

    def _measure_azimuth(self, audio, sr):
        fl = np.fft.rfft(audio[0])
        fr = np.fft.rfft(audio[1])
        fq = np.fft.rfftfreq(len(audio[0]), d=1.0 / sr)
        hf = fq >= 8000
        if not np.any(hf):
            return 0.0
        return float(np.degrees(np.median(np.angle(fr[hf]) - np.angle(fl[hf]))))

    def _fix_azimuth(self, audio, sr, deg):
        result = np.asarray(audio, dtype=np.float32).copy()
        fr = np.fft.rfft(result[1])
        fq = np.fft.rfftfreq(len(result[1]), d=1.0 / sr)
        hf = fq >= 8000
        if np.any(hf):
            fr[hf] *= np.exp(-1j * np.radians(deg) * 0.5)
            result[1] = np.fft.irfft(fr, n=len(result[1]))[: len(result[1])]
        return result

    def _merge_regions(self, regions, max_gap_ms, sr):
        if len(regions) < 2:
            return regions
        max_gap = int(sr * max_gap_ms / 1000)
        s = sorted(regions, key=lambda x: x[0])
        merged = [s[0]]
        for r in s[1:]:
            if r[0] <= merged[-1][1] + max_gap:
                a, b = merged[-1][0], merged[-1][1]
                c_val = merged[-1][2] if len(merged[-1]) > 2 else 0
                merged[-1] = (a, max(b, r[1]), min(c_val, r[2]) if len(r) > 2 else c_val)
            else:
                merged.append(r)
        return merged
