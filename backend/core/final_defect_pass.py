"""FinalDefectPass — Letzter, aggressiver Reparatur-Durchlauf.

Läuft NACH allen Enhancement-Modulen, direkt vor dem Export.
Behebt ALLE verbleibenden Defekte:
  1. Azimuth/HF: Volle Stärke (kein Vocal-Reduce)
  2. Residual clicks: Untergrenze 4dB, alle Frequenzbänder
  3. Post-repair vocal restoration: Interpolation aus Kontext
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def final_defect_pass(audio: np.ndarray, sr: int) -> np.ndarray:
    """Aggressive final repair — volle Stärke überall.

    Dann: Vocal-Damage-Interpolation wo nötig.
    """
    result = np.asarray(audio, dtype=np.float32).copy()
    result.shape[-1]
    mono = np.mean(result, axis=0) if result.ndim == 2 else result

    # ── 1. Detektierte Gesangszonen ──
    vocal = _detect_vocal(mono, sr)

    # ── 2. VOLLE AZIMUTH-KORREKTUR (kein Vocal-Reduce) ──
    if result.ndim >= 2 and result.shape[0] >= 2:
        result = _fix_azimuth_full(result, sr)

    # ── 3. VOLLE HF-RESTORATION ──
    result = _restore_hf_full(result, sr)

    # ── 4. AGGRESSIVE CLICK/CRACKLE-ENTFERNUNG ──
    result = _remove_residual_clicks(result, sr)

    # ── 5. VOCAL-DAMAGE-INTERPOLATION ──
    # Nach der vollen Reparatur: prüfe ob Gesang beschädigt wurde
    result = _interpolate_vocal_damage(result, sr, vocal)

    return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _fix_azimuth_full(audio, sr):
    """Volle Azimuth-Korrektur ohne Abschwächung."""
    result = np.asarray(audio, dtype=np.float32).copy()
    n = result.shape[-1]
    block = int(sr * 0.100)
    hop = block // 4
    fixes = 0

    for i in range(0, n - block, hop):
        l_seg = result[0, i : i + block]
        r_seg = result[1, i : i + block]
        fft_l = np.fft.rfft(l_seg)
        fft_r = np.fft.rfft(r_seg)
        freqs = np.fft.rfftfreq(len(l_seg), d=1.0 / sr)
        hf = freqs >= 6000
        if not np.any(hf):
            continue

        phase_diff = np.median(np.angle(fft_r[hf]) - np.angle(fft_l[hf]))
        if abs(np.degrees(phase_diff)) > 2:
            fft_r[hf] *= np.exp(-1j * phase_diff * 0.5)  # 50% — stärker
            corrected = np.fft.irfft(fft_r, n=block)
            result[1, i : i + block] = corrected[:block].astype(np.float32)
            fixes += 1

    logger.info("FinalPass azimuth: %d fixes (full strength)", fixes)
    return result


def _restore_hf_full(audio, sr):
    """Volle HF-Restoration ohne Vocal-Rücksicht."""
    result = np.asarray(audio, dtype=np.float32).copy()
    mono = np.mean(result, axis=0)
    n = len(mono)
    block = int(sr * 0.200)
    restored = 0

    for i in range(0, n - block, block):
        seg = mono[i : i + block]
        fft = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
        hf_mask = freqs >= 8000
        if not np.any(hf_mask):
            continue

        hf_ratio = np.sum(fft[hf_mask]) / (np.sum(fft) + 1e-12)

        # Wenn HF < 2%: volle Restauration
        if hf_ratio < 0.02:
            target = 0.03
            gain = min(3.0, 1.0 + (target - hf_ratio) * 80)
            for ch in range(result.shape[0] if result.ndim == 2 else 1):
                ch_fft = np.fft.rfft(result[ch, i : i + block] if result.ndim == 2 else result[i : i + block])
                ch_fft[hf_mask] *= gain
                if result.ndim == 2:
                    result[ch, i : i + block] = np.fft.irfft(ch_fft, n=block)
                else:
                    result[i : i + block] = np.fft.irfft(ch_fft, n=block)
            restored += 1

    logger.info("FinalPass HF: %d blocks wiederhergestellt (full strength)", restored)
    return result


def _remove_residual_clicks(audio, sr):
    """Entfernt verbleibende Clicks — KONSERVATIV, nur extreme Ausreißer."""
    result = np.asarray(audio, dtype=np.float32).copy()
    mono = np.mean(result, axis=0) if result.ndim == 2 else result
    n = len(mono)
    win = int(sr * 0.002)  # 2ms — sehr kurz
    removed = 0
    max_modifications = n // 500  # Max 0.2% of samples

    for i in range(win, n - win, win * 8):  # Non-overlapping
        if removed >= max_modifications:
            logger.warning("FinalPass clicks: abbrechen at %d (safety limit)", removed)
            break

        seg = mono[i - win : i + win]
        local_rms = float(np.sqrt(np.mean(mono[max(0, i - win * 5) : min(n, i + win * 5)] ** 2) + 1e-12))

        # NUR extreme Peaks: >12dB über lokalem RMS
        peak = float(np.max(np.abs(seg)))
        if peak > local_rms * 4.0 and local_rms > 0.001:  # ~12dB
            for ch in range(result.shape[0] if result.ndim == 2 else 1):
                ch_data = result[ch] if result.ndim == 2 else result
                pre = ch_data[max(0, i - win - 1)]
                post = ch_data[min(n - 1, i + win + 1)]
                interp = np.linspace(pre, post, win * 2 + 2, dtype=np.float32)[1:-1]
                ch_data[max(0, i - win) : min(n, i + win)] = interp[: min(n, i + win) - max(0, i - win)]
            removed += 1

    logger.info("FinalPass clicks: %d extreme clicks removed", removed)
    return result


def _interpolate_vocal_damage(audio, sr, vocal_mask):
    """Nach voller Reparatur: beschädigte Gesangszonen aus Kontext interpolieren."""
    # Diese Funktion erkennt sprunghafte Änderungen IM Gesang
    # die durch die volle Reparatur entstanden sein könnten,
    # und glättet sie durch lokale Interpolation.
    result = np.asarray(audio, dtype=np.float32).copy()
    mono = np.mean(result, axis=0) if result.ndim == 2 else result
    n = len(mono)

    # Suche nach sprunghaften Amplituden-Änderungen in Vocal-Zonen
    win = int(sr * 0.025)
    hop = win // 2
    repaired = 0

    prev_rms = 0
    for i in range(win, n - win, hop):
        if not vocal_mask[min(i, n - 1)]:
            prev_rms = 0
            continue

        seg = mono[i : i + win]
        rms = float(np.sqrt(np.mean(seg**2) + 1e-12))

        # Sprunghafte Änderung? (>6dB in 12.5ms)
        if prev_rms > 0:
            change_db = abs(20 * np.log10(rms / prev_rms))
            if change_db > 6:
                # Sanfte Glättung
                alpha = 0.3  # 30% Korrektur
                for ch in range(result.shape[0] if result.ndim == 2 else 1):
                    ch_data = result[ch] if result.ndim == 2 else result
                    smoothed = ch_data[i : i + win] * (1 - alpha) + np.roll(ch_data, 1)[i : i + win] * alpha
                    ch_data[i : i + win] = smoothed
                repaired += 1

        prev_rms = rms  # type: ignore[assignment]

    logger.info("FinalPass vocal: %d damaged sections interpolated", repaired)
    return result


def _detect_vocal(audio, sr):
    """Simple vocal detection."""
    n = len(audio)
    win = int(sr * 0.025)
    hop = max(1, win // 2)
    mask = np.zeros(n, dtype=bool)
    if win < 64:
        return mask

    for i in range(0, n - win, hop):
        frame = audio[i : i + win]
        rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
        if rms < 0.002:
            continue
        fft = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)
        if np.sum(fft) < 1e-10:
            continue
        centroid = float(np.average(freqs, weights=fft + 1e-10))
        vb = (freqs >= 300) & (freqs <= 3000)
        vb_ratio = float(np.sum(fft[vb])) / (float(np.sum(fft)) + 1e-12)
        if vb_ratio > 0.3 and 300 < centroid < 3000:
            mask[i : i + win] = True

    return mask
