"""Precision Dropout Repair — nur echte Bandkopf-Defekte, kein Over-Repair.

Nutzt statistische Ausreißer-Erkennung (MAD-basiert) statt einfacher
Schwellwerte. Repariert NUR Samples, die signifikant von der lokalen
Signalverteilung abweichen — musikalische Dynamik bleibt unangetastet.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def repair_dropouts_precise(audio, sr):
    """Präzise Dropout-Reparatur via MAD-Ausreißer-Erkennung.

    Returns: (repaired_audio, num_dropouts_found, num_dropouts_repaired)
    """
    result = np.asarray(audio, dtype=np.float32).copy()
    mono = np.mean(result, axis=0) if result.ndim == 2 else result
    n = len(mono)

    # ── 1. Lokale RMS-Hüllkurve (10ms Fenster, 5ms Hop) ──
    rms_win = int(sr * 0.010)
    hop = rms_win // 2
    rms_env = []
    for i in range(0, n - rms_win, hop):
        seg = mono[i : i + rms_win]
        rms_env.append(float(np.sqrt(np.mean(seg**2) + 1e-12)))
    rms_env = np.array(rms_env, dtype=np.float64)  # type: ignore[assignment]

    # ── 2. Gleitende MAD-basierte Ausreißer-Erkennung ──
    # Für jeden Punkt: vergleiche mit lokaler Verteilung (100 Punkte = 500ms)
    local_n = 100
    threshold_sigma = 3.5  # Sensitiver: 3.5σ

    outliers = []
    for i in range(local_n, len(rms_env) - local_n):
        local = rms_env[i - local_n : i + local_n]
        median = float(np.median(local))
        mad = float(np.median(np.abs(local - median))) * 1.4826  # type: ignore  # Skalierung für Normalverteilung
        if mad < 1e-12:
            continue

        # Z-Score basierend auf MAD
        z_score = (rms_env[i] - median) / mad

        # Nur negative Ausreißer (Pegel-Einbrüche)
        if z_score < -threshold_sigma:
            t_sample = i * hop
            # Finde die genauen Grenzen des Einbruchs
            s0 = max(0, t_sample - rms_win)
            s1 = min(n, t_sample + rms_win)

            # Verifiziere: ist der Einbruch kurz (<30ms)?
            duration_ms = (s1 - s0) / sr * 1000
            if 2 <= duration_ms <= 30:
                outliers.append((s0, s1, z_score))

    # ── 3. Überlappende Regionen mergen ──
    if len(outliers) > 1:
        outliers.sort(key=lambda x: x[0])
        merged = [outliers[0]]
        for o in outliers[1:]:
            if o[0] <= merged[-1][1] + rms_win:
                merged[-1] = (merged[-1][0], max(merged[-1][1], o[1]), min(merged[-1][2], o[2]))
            else:
                merged.append(o)
        outliers = merged

    # ── 4. Chirurgische Reparatur ──
    repaired = 0
    for s0, s1, z_score in outliers:
        # Nur reparieren wenn der Einbruch substanziell ist (mind. 4dB)
        seg_orig = mono[s0:s1]
        seg_rms = float(np.sqrt(np.mean(seg_orig**2) + 1e-12))
        ctx_before = mono[max(0, s0 - rms_win) : s0]
        ctx_after = mono[s1 : min(n, s1 + rms_win)]
        ctx_rms = float(np.sqrt(np.mean(np.concatenate([ctx_before, ctx_after]) ** 2) + 1e-12))

        drop_db = 20 * np.log10(seg_rms / ctx_rms) if ctx_rms > 0 else 0
        if drop_db > -3:  # Weniger als 3dB = kein echter Dropout
            continue

        # Reparatur: nur diesen Bereich, kubische Interpolation
        length = s1 - s0
        if length < 3:
            continue

        try:
            for ch in range(result.shape[0] if result.ndim == 2 else 1):
                ch_data = result[ch] if result.ndim == 2 else result

                # 4 Stützpunkte für sanfte Interpolation
                x_pts = [max(0, s0 - 4), s0 + length // 3, s0 + 2 * length // 3, min(n - 1, s1 + 4)]
                y_vals = [ch_data[min(n - 1, max(0, x))] for x in x_pts]

                coeffs = np.polyfit(x_pts, y_vals, min(2, len(x_pts) - 1))
                xi = np.arange(s0, s1)
                yi = np.polyval(coeffs, xi)

                # Cross-fade (1ms) für unhörbaren Übergang
                xf = min(int(sr * 0.001), length // 3)
                if xf >= 2:
                    w = np.ones(length)
                    w[:xf] = np.linspace(0, 1, xf)
                    w[-xf:] = np.linspace(1, 0, xf)
                    ch_data[s0:s1] = yi.astype(np.float32) * w + ch_data[s0:s1] * (1 - w)
                else:
                    ch_data[s0:s1] = yi.astype(np.float32)

                if result.ndim == 2:
                    result[ch] = ch_data[: len(result[ch])]
                else:
                    result = ch_data[: len(result)].astype(np.float32)

            repaired += 1
        except Exception as e:
            logger.warning("precision_dropout_repair.py::unbekannter Ersatzpfad: %s", e)

    return np.clip(result, -1.0, 1.0).astype(np.float32), len(outliers), repaired
