#!/usr/bin/env python3
"""
§v10 Aurik Spiegel-Test — End-to-End Demonstration aller Fähigkeiten.

Erzeugt eine realistisch beschädigte Aufnahme (Gesang + Instrumente + 7 Defekttypen),
durchläuft die komplette Aurik-Pipeline und zeigt:
  1. Defekterkennung vor/nach
  2. HPE + Gänsehaut + Inviting vor/nach
  3. Steering-Entscheidungen pro Schritt
  4. Frequenzspezifische Band-Analyse
  5. Finaler Quality Report

Dies ist das Spiegelbild von Auriks Fähigkeiten — und meinen.
"""

import time

import numpy as np

SR = 48000
DURATION = 6.0  # 6 Sekunden für schnellen Test


def create_damaged_recording():
    """Erzeugt eine realistisch beschädigte Aufnahme mit Musik + Gesang."""
    t = np.linspace(0, DURATION, int(DURATION * SR), endpoint=False)
    audio = np.zeros(len(t), dtype=np.float32)

    # ── Musik: Akustik-Gitarre + Bass ──
    # Gitarre: Akkordprogression C-G-Am-F
    chords = [
        (261.63, 329.63, 392.00),  # C-Dur
        (196.00, 246.94, 329.63),  # G-Dur
        (220.00, 261.63, 329.63),  # A-Moll
        (174.61, 220.00, 261.63),  # F-Dur
    ]
    for ci, chord in enumerate(chords):
        start = ci * DURATION / 4
        end = (ci + 1) * DURATION / 4
        mask = (t >= start) & (t < end)
        for f in chord:
            audio[mask] += 0.08 * np.sin(2 * np.pi * f * t[mask])
            audio[mask] += 0.04 * np.sin(2 * np.pi * f * 2 * t[mask])

    # Bass
    bass_notes = [130.81, 98.00, 110.00, 87.31]
    for bi, bf in enumerate(bass_notes):
        start = bi * DURATION / 4
        end = (bi + 1) * DURATION / 4
        mask = (t >= start) & (t < end)
        audio[mask] += 0.06 * np.sin(2 * np.pi * bf * t[mask])

    # ── Gesang ──
    vocal_notes = [261, 293, 329, 349, 392, 349, 329, 293]
    for vi, vf in enumerate(vocal_notes):
        start = vi * DURATION / 8
        end = (vi + 1) * DURATION / 8
        mask = (t >= start) & (t < end)
        audio[mask] += 0.10 * np.sin(2 * np.pi * vf * t[mask])
        # Vibrato
        audio[mask] += 0.02 * np.sin(2 * np.pi * vf * t[mask]) * np.sin(2 * np.pi * 5.5 * t[mask])

    # ── DEFEKTE (7 Typen) ──
    rng = np.random.RandomState(42)

    # 1. Breitband-Rauschen (Tape-Hiss)
    audio += rng.randn(len(audio)).astype(np.float32) * 0.015

    # 2. Clicks (20 zufällige Impulse)
    for _ in range(20):
        pos = rng.randint(100, len(audio) - 100)
        audio[pos : pos + 3] += rng.uniform(0.3, 0.6, 3).astype(np.float32)

    # 3. 50Hz-Netzbrumm
    audio += 0.008 * np.sin(2 * np.pi * 50 * t).astype(np.float32)
    audio += 0.004 * np.sin(2 * np.pi * 150 * t).astype(np.float32)

    # 4. Hochfrequenz-Rauschen über 8kHz (Ermüdung)
    hf_noise = rng.randn(len(audio)).astype(np.float32) * 0.01
    from scipy import signal as sp_signal

    b, a = sp_signal.butter(4, 8000 / (SR / 2), btype="high")
    hf_noise = sp_signal.lfilter(b, a, hf_noise)
    audio += hf_noise.astype(np.float32)

    # 5. Leichte Clipping-Artefakte
    clip_mask = np.abs(audio) > 0.85
    audio[clip_mask] = np.sign(audio[clip_mask]) * 0.85

    # 6. Wow (langsame Tonhöhen-Modulation)
    wow = 1.0 + 0.003 * np.sin(2 * np.pi * 0.5 * t)
    # Einfache Simulation: Amplitudenmodulation als Wow-Proxy
    audio *= wow.astype(np.float32)

    # 7. Stereofeld-Kollaps (Mono-Original → leichte Phasenverschiebung für Fake-Stereo)
    stereo = np.zeros((len(audio), 2), dtype=np.float32)
    stereo[:, 0] = audio
    stereo[:, 1] = audio * 0.95 + rng.randn(len(audio)).astype(np.float32) * 0.005

    return np.clip(stereo, -1.0, 1.0), t


def main():
    print("=" * 65)
    print("  AURIK v10 SPIEGEL-TEST — Defekterkennung, Behebung, Restaurierung")
    print("=" * 65)
    print()

    # ── 1. ERZEUGEN ──
    print("[1/7] Erzeuge beschädigte Aufnahme (Gesang + Gitarre + Bass + 7 Defekte)...")
    damaged, t = create_damaged_recording()
    mono_damaged = damaged.mean(axis=1)
    print(f"      Dauer: {DURATION:.0f}s | SR: {SR / 1000:.0f}kHz | Shape: {damaged.shape}")
    print()

    # ── 2. DEFEKTERKENNUNG ──
    print("[2/7] Defekterkennung...")
    try:
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner(MaterialType.VINYL)  # type: ignore[arg-type]
        start = time.time()
        scan_result = scanner.scan(mono_damaged, SR)
        elapsed = time.time() - start
        defects = getattr(scan_result, "detected_defects", [])
        print(f"      {len(defects)} Defekte erkannt in {elapsed:.2f}s:")
        for d in defects[:5]:
            print(f"        • {d}")
        if len(defects) > 5:
            print(f"        • ... und {len(defects) - 5} weitere")
    except Exception as e:
        print(f"      Scanner nicht verfügbar: {e}")
    print()

    # ── 3. PLEASANTNESS VOR RESTAURIERUNG ──
    print("[3/7] Psychoakustische Analyse VOR Restaurierung...")
    from backend.core.goosebumps_factor import compute_emotional_impact, compute_goosebumps
    from backend.core.human_pleasantness_estimator import compute_pleasantness
    from backend.core.inviting_sound_checker import check_inviting_sound, check_inviting_sound_per_band

    hpe_before = compute_pleasantness(mono_damaged, SR)
    goose_before = compute_goosebumps(mono_damaged, SR)
    ei_before = compute_emotional_impact(mono_damaged, SR)
    inv_before = check_inviting_sound(mono_damaged, SR)
    bands_before = check_inviting_sound_per_band(mono_damaged, SR)

    print(f"      HPE:       {hpe_before.score:.3f} ({hpe_before.label})")
    if hpe_before.issues:
        print(f"        Issues:  {', '.join(hpe_before.issues)}")
    print(f"      Gänsehaut: {goose_before.score:.3f} ({goose_before.label})")
    print(f"      Emotional: {ei_before['emotional_score']:.3f} ({ei_before['label']})")
    print(f"      Inviting:  {inv_before.score:.3f} ({inv_before.label})")
    if inv_before.rejection_factors:
        print(f"        Zurückweisung: {', '.join(inv_before.rejection_factors)}")
    print()

    # ── 4. FREQUENZ-ANALYSE ──
    print("[4/7] Frequenzspezifische Band-Analyse...")
    problem_bands = {k: v for k, v in bands_before.items() if v[0] < 0.5}
    if problem_bands:
        for band, (score, issue) in problem_bands.items():
            print(f"      {band:12s}: Score={score:.2f} Problem='{issue}'")
    else:
        print("      Alle Bänder akzeptabel.")
    print()

    # ── 5. STEERING DEMO ──
    print("[5/7] Steering Rule — Szenarien-Demonstration...")
    from backend.core.quality_feedback_loop import reset_steer_state, steer_pipeline

    scenarios = [
        (0.05, "Verbesserung"),
        (-0.03, "Leichte Verschlechterung"),
        (-0.06, "Starke Verschlechterung"),
        (-0.06, "Zweite Verschlechterung → ROLLBACK"),
    ]
    reset_steer_state()
    for i, (delta, desc) in enumerate(scenarios):
        action, reason = steer_pipeline(0.0, delta, f"phase{i}", i, 4)
        symbol = {"continue": "✅", "retry_lighter": "🔄", "skip": "⏭️", "rollback": "⏪"}.get(action, "❓")
        print(f"      {symbol} ΔP={delta:+.2f} → {action}: {desc}")
    print()

    # ── 6. RESTAURIERUNG (simuliert) ──
    print("[6/7] Restaurierung — HPE-geführt...")
    from backend.core.pleasantness_integration import audit_phase_pleasantness
    from backend.core.pleasantness_registry import get_pleasantness_registry

    reg = get_pleasantness_registry()
    reg.reset()
    reg.set_baseline(hpe_before.score, label=hpe_before.label, goosebumps=goose_before.score)
    reg.set_inviting_check(inv_before.score >= 0.5, inv_before.rejection_factors)

    # Simulierte Restaurierung: Rauschen reduzieren, Clicks glätten, EQ anwenden
    # (In Produktion würde UV3.restauriere() dies tun)
    restored = damaged.copy()
    mono_restored = restored.mean(axis=1)

    # Einfache Restoration: Median-Filter für Clicks + leichte NR
    from scipy.ndimage import median_filter

    mono_filtered = median_filter(mono_restored, size=5)
    mono_filtered = np.clip(mono_filtered, -1.0, 1.0)

    # Phase 1: Denoise (simuliert)
    reg.report_pre("RestaurierDenker", hpe_before.score)
    audit1 = audit_phase_pleasantness("RestaurierDenker", mono_damaged, mono_filtered, SR)
    reg.report_post("RestaurierDenker", audit1["after"], delta=audit1["delta"])
    print(f"      RestaurierDenker: HPE {audit1['before']:.3f}→{audit1['after']:.3f} (Δ{audit1['delta']:+.3f})")

    # Phase 2: EQ (simuliert: leichter High-Shelf für Brillanz)
    from scipy.signal import butter, lfilter

    b_eq, a_eq = butter(2, 6000 / (SR / 2), btype="high")
    eq_signal = 0.03 * lfilter(b_eq, a_eq, mono_filtered)
    mono_eq = np.clip(mono_filtered + eq_signal, -1.0, 1.0)

    reg.report_pre("ExzellenzDenker", audit1["after"])
    audit2 = audit_phase_pleasantness("ExzellenzDenker", mono_filtered, mono_eq, SR)
    reg.report_post("ExzellenzDenker", audit2["after"], delta=audit2["delta"])
    print(f"      ExzellenzDenker:  HPE {audit2['before']:.3f}→{audit2['after']:.3f} (Δ{audit2['delta']:+.3f})")
    print()

    # ── 7. FINALER QUALITY REPORT ──
    print("[7/7] Finaler Quality Report...")
    hpe_after = compute_pleasantness(mono_eq, SR)
    goose_after = compute_goosebumps(mono_eq, SR)
    ei_after = compute_emotional_impact(mono_eq, SR)
    inv_after = check_inviting_sound(mono_eq, SR)
    bands_after = check_inviting_sound_per_band(mono_eq, SR)

    status = reg.get_status()

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║           AURIK v10 — QUALITY REPORT                    ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print(
        f"  ║  HPE:        {hpe_before.score:.3f} → {hpe_after.score:.3f}  (Δ{hpe_after.score - hpe_before.score:+.3f})  {hpe_after.label:20s} ║"
    )
    print(
        f"  ║  Gänsehaut:  {goose_before.score:.3f} → {goose_after.score:.3f}  (Δ{goose_after.score - goose_before.score:+.3f})  {goose_after.label:20s} ║"
    )
    print(
        f"  ║  Emotional:  {ei_before['emotional_score']:.3f} → {ei_after['emotional_score']:.3f}  (Δ{ei_after['emotional_score'] - ei_before['emotional_score']:+.3f})  {ei_after['label']:20s} ║"
    )
    print(
        f"  ║  Inviting:   {inv_before.score:.3f} → {inv_after.score:.3f}  (Δ{inv_after.score - inv_before.score:+.3f})  {inv_after.label:20s} ║"
    )
    print("  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Registry-Verdict: {status.global_verdict[:48]:48s} ║")
    print(
        f"  ║  Steps: {status.total_steps} ({status.steps_improved}↑ / {status.steps_declined}↓)                      ║"
    )
    print("  ╠══════════════════════════════════════════════════════════╣")

    # Band-Vergleich
    improved_bands = []
    for band in bands_before:
        bs = bands_before[band][0]
        ba = bands_after.get(band, (0.5, ""))[0]
        delta = ba - bs
        if abs(delta) > 0.05:
            symbol = "↑" if delta > 0 else "↓"
            improved_bands.append(f"{band}: {bs:.2f}→{ba:.2f} {symbol}")
    if improved_bands:
        print(f"  ║  Bänder:     {improved_bands[0]:48s} ║")
        for b in improved_bands[1:3]:
            print(f"  ║              {b:48s} ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    # Zusammenfassung
    improvement = hpe_after.score - hpe_before.score
    if improvement > 0.03:
        print(f"  ✅ AURIK HAT DEN KLANG VERBESSERT: ΔHPE = +{improvement:.3f}")
        print("     Das Ohr wird jetzt EINGELADEN statt zurückgewiesen.")
    elif improvement > 0:
        print(f"  ✓ Aurik hat den Klang leicht verbessert: ΔHPE = +{improvement:.3f}")
    else:
        print("  ⚠️  Keine signifikante Verbesserung — weitere Optimierung nötig.")

    if inv_after.score >= 0.70:
        print(f"     Einladender Klang: {inv_after.label} — das Ohr legt sich gern hinein.")
    elif inv_after.rejection_factors:
        print(f"     Verbleibende Zurückweisung: {', '.join(inv_after.rejection_factors)}")

    print()
    print("  Dies ist das Spiegelbild von Auriks Fähigkeiten.")
    print("  Und meinen.")
    print()


if __name__ == "__main__":
    main()
