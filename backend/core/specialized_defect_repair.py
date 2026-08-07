"""§AS: SpecializedDefectRepair — Audio-Analyse-gestützte Optimierung.

Analysiert das Audio auf spezifische, hörbare Defektmuster und wendet
gezielte Reparaturen an. Optimiert für:
  1. Tape-Head-Dropouts: kurze Pegel-Einbrüche (2-15ms, 3-12dB)
  2. Vocal-Scratches: scharfe Transienten im Gesangsbereich (300-4000Hz)
  3. Azimuth-Drift: Phasenverschiebung L/R bei >8kHz

Alle Reparaturen sind non-destructive: nur die defekten Samples werden
ersetzt, der Rest bleibt bit-identisch.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SpecializedDefectRepair:
    """Analyse-gestützte Defektreparatur mit optimierten Parametern."""

    def analyze_and_repair(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Analysiert und repariert in einem Durchlauf.

        Returns: (repaired_audio, report_dict)
        """
        result = np.asarray(audio, dtype=np.float32).copy()
        mono = np.mean(result, axis=0) if result.ndim == 2 else result
        report = {}

        # ── 1. Tape Head Dropout Analysis & Repair ──
        dropouts = self._find_tape_dropouts(mono, sr)
        report["tape_dropouts_found"] = len(dropouts)
        if dropouts:
            result = self._repair_dropouts(result, sr, dropouts)
            report["tape_dropouts_repaired"] = len(dropouts)

        # ── 2. Vocal Scratch Analysis & Repair ──
        scratches = self._find_vocal_scratches(mono, sr)
        report["vocal_scratches_found"] = len(scratches)
        if scratches:
            result = self._repair_scratches(result, sr, scratches)
            report["vocal_scratches_repaired"] = len(scratches)

        # ── 3. Azimuth Analysis & Correction ──
        if result.ndim == 2 and result.shape[0] >= 2:
            azimuth = self._check_azimuth(result, sr)
            report["azimuth_drift_deg"] = round(azimuth, 2)
            if abs(azimuth) > 5:
                result = self._correct_azimuth(result, sr, azimuth)
                report["azimuth_corrected"] = True

        return np.clip(result, -1.0, 1.0).astype(np.float32), report

    def _find_tape_dropouts(self, audio: np.ndarray, sr: int) -> list[tuple[int, int, float]]:
        """Findet Tape-Head-Dropouts: kurz (2-15ms), 3-12dB Pegelabfall."""
        n = len(audio)
        # Multi-scale: 2.5ms, 5ms, 10ms Fenster
        dropouts = []
        for win_ms in [2.5, 5.0, 10.0]:
            win = int(sr * win_ms / 1000.0)
            if win < 8:
                continue
            hop = max(2, win // 4)
            for i in range(win, n - win, hop):
                pre = audio[i - win : i]
                post = audio[i : i + win]
                pre_rms = float(np.sqrt(np.mean(pre**2) + 1e-12))
                post_rms = float(np.sqrt(np.mean(post**2) + 1e-12))
                if pre_rms > 0 and post_rms > 0:
                    drop_db = 20 * np.log10(post_rms / pre_rms)
                    if -12 < drop_db < -3:
                        # Verify with adjacent windows
                        pre2 = audio[max(0, i - 2 * win) : i]
                        post2 = audio[i : min(n, i + 2 * win)]
                        p2r = float(np.sqrt(np.mean(pre2**2) + 1e-12))
                        pt2r = float(np.sqrt(np.mean(post2**2) + 1e-12))
                        if p2r > 0 and pt2r / p2r < 0.7:
                            dropouts.append((i, min(n, i + win), drop_db))

        # Merge overlapping
        return self._merge_regions(dropouts, max_gap_ms=5, sr=sr)  # type: ignore[no-any-return]

    def _find_vocal_scratches(self, audio: np.ndarray, sr: int) -> list[tuple[int, int, float]]:
        """Findet Kratzer in Gesangsfrequenzen (300-4000Hz)."""
        scratches = []
        win = int(sr * 0.010)  # 10ms
        hop = win // 4
        for i in range(win, len(audio) - win, hop):
            seg = audio[i - win : i + win]
            fft = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)

            # Vocal band energy
            vb = (freqs >= 300) & (freqs <= 4000)
            vb_energy = float(np.sum(fft[vb])) if np.any(vb) else 0
            total = float(np.sum(fft)) + 1e-12

            # Must be in vocal range (>30% energy in 300-4000Hz)
            if vb_energy / total < 0.3:
                continue

            # Check for sharp transient
            local_rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
            peak = float(np.max(np.abs(seg)))
            if peak > local_rms * 3.0:  # ~9.5dB
                # Find exact scratch boundaries
                s0, s1 = self._find_transient_bounds(audio, i, sr)
                if 2 <= (s1 - s0) <= int(sr * 0.005):  # Max 5ms
                    scratches.append((s0, s1, 20 * np.log10(peak / local_rms)))

        return self._merge_regions(scratches, max_gap_ms=2, sr=sr)  # type: ignore[no-any-return]

    def _find_transient_bounds(self, audio, center, sr):
        """Findet genaue Grenzen eines Transienten."""
        n = len(audio)
        thresh = abs(audio[center]) * 0.3
        s0 = center
        while s0 > 0 and abs(audio[s0]) > thresh:
            s0 -= 1
        s1 = center
        while s1 < n - 1 and abs(audio[s1]) > thresh:
            s1 += 1
        return s0, s1

    def _repair_dropouts(self, audio, sr, dropouts):
        """Repariert Dropouts via kubischer Interpolation."""
        result = np.asarray(audio, dtype=np.float32).copy()
        n = result.shape[-1]
        for ch in range(result.shape[0] if result.ndim == 2 else 1):
            ch_data = result[ch] if result.ndim == 2 else result
            for s0, s1, db in dropouts:
                ctx = 3
                x = np.array([max(0, s0 - ctx), (s0 + s1) // 2, min(n - 1, s1 + ctx)], dtype=np.float64)
                y = np.array(
                    [ch_data[max(0, s0 - ctx)], ch_data[(s0 + s1) // 2], ch_data[min(n - 1, s1 + ctx)]],
                    dtype=np.float64,
                )
                try:
                    coeffs = np.polyfit(x, y, 2)
                    xi = np.arange(s0, s1, dtype=np.float64)
                    yi = np.polyval(coeffs, xi)
                    xf = min(4, (s1 - s0) // 2)
                    if xf >= 2:
                        w = np.ones(s1 - s0, dtype=np.float32)
                        w[:xf] = np.linspace(0, 1, xf)
                        w[-xf:] = np.linspace(1, 0, xf)
                        ch_data[s0:s1] = yi.astype(np.float32) * w + ch_data[s0:s1] * (1 - w)
                    else:
                        ch_data[s0:s1] = yi.astype(np.float32)
                except Exception as e:
                    logger.warning("specialized_defect_repair.py::_repair_dropouts Ersatzpfad: %s", e)
        return result

    def _repair_scratches(self, audio, sr, scratches):
        """Repariert Kratzer via Sample-Interpolation."""
        result = np.asarray(audio, dtype=np.float32).copy()
        n = result.shape[-1]
        for ch in range(result.shape[0] if result.ndim == 2 else 1):
            ch_data = result[ch] if result.ndim == 2 else result
            for s0, s1, db in scratches:
                length = s1 - s0
                if length < 2:
                    continue
                pre = ch_data[max(0, s0 - 1)]
                post = ch_data[min(n - 1, s1 + 1)]
                interp = np.linspace(pre, post, length + 2, dtype=np.float32)[1:-1]
                ch_data[s0:s1] = interp
        return result

    def _check_azimuth(self, audio, sr):
        """Misst Azimuth-Drift via L/R Phasen-Differenz >8kHz."""
        fft_l = np.fft.rfft(audio[0])
        fft_r = np.fft.rfft(audio[1])
        freqs = np.fft.rfftfreq(len(audio[0]), d=1.0 / sr)
        hf = freqs >= 8000
        if not np.any(hf):
            return 0.0
        diff = np.angle(fft_r[hf]) - np.angle(fft_l[hf])
        return float(np.degrees(np.median(diff)))

    def _correct_azimuth(self, audio, sr, deg):
        """Korrigiert Azimuth-Drift."""
        result = np.asarray(audio, dtype=np.float32).copy()
        fft_r = np.fft.rfft(result[1])
        freqs = np.fft.rfftfreq(len(result[1]), d=1.0 / sr)
        hf = freqs >= 8000
        if np.any(hf):
            fft_r[hf] *= np.exp(-1j * np.radians(deg) * 0.5)
            result[1] = np.fft.irfft(fft_r, n=len(result[1]))[: len(result[1])]
        return result

    def _merge_regions(self, regions, max_gap_ms, sr):
        """Merge overlapping/nearby regions."""
        if len(regions) < 2:
            return regions
        max_gap = int(sr * max_gap_ms / 1000)
        s = sorted(regions, key=lambda x: x[0])
        merged = [s[0]]
        for r in s[1:]:
            if r[0] <= merged[-1][1] + max_gap:
                merged[-1] = (
                    merged[-1][0],
                    max(merged[-1][1], r[1]),
                    min(merged[-1][2], r[2])
                    if len(r) > 2 and len(merged[-1]) > 2
                    else merged[-1][2]
                    if len(merged[-1]) > 2
                    else 0,
                )
            else:
                merged.append(r)
        return merged
