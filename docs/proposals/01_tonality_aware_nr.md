# Vorschlag 01 — Tonalitäts-Gate für alle NR-Module

> **Status: VORSCHLAG (Entwurf) — nicht normativ.** Übernahme erst nach
> Maintainer-Sign-off als neue Spec `23_tonality_aware_nr.md`.
> Formel und Meta-Regel: `00_interpretationsfrei_uebersicht.md`.

## Zweck

Stationäre tonale Signale (Musik, Sinusanteile) werden von
Rauschschätzern systematisch als Rauschen klassifiziert (5 %-Perzentil-
Rauschboden, IMCRA). Mess-Evidenz aus der Test-Session:
spectral flatness **0.00** (Sinus 440+880 Hz), **0.69** (Sinus + Rauschen
σ=0.05), **0.99** (reines Rauschen). Das Gate verhindert, dass NR-Module
Musik wegdämpfen — als Garant, nicht als Einzelfall-Logik.

## Normativer Wortlaut (für Spec 23)

> `[RELEASE_MUST] §23-TONALITY: Jedes NR-Modul — mindestens
> `backend/core/phases/phase_03_denoise.py`, `dsp/spectral_denoiser.py`,
> `backend/core/hybrid/hybrid_ml_denoiser.py` sowie jeder SGMSE+/DFN-
> Einsatzpunkt in `backend/core/phases/` — MUSS vor der ML-NR-Entscheidung
> `backend.core.dsp.tonality_gate.is_tonal_clean(audio, sr)` konsultieren.
> Rückgabe `True` ⇒ ML-NR-Zweig überspringen; der DSP-Pfad bleibt aktiv.
> Definition: spectral flatness auf Welch-Spektrum (hann,
> nperseg=min(2048, n//2), f ≥ 100 Hz) < `TONAL_CLEAN_FLATNESS`.
> Die Konstante `TONAL_CLEAN_FLATNESS = 0.05` liegt ausschließlich in
> `backend/core/calibrated_constants.py`. Das Ergebnis der Konsultation
> MUSS in `metadata["tonality_gate"]` protokolliert werden
> (`true`/`false`/`skipped`).

## Enforcement

1. **Lint (fail-closed)** in `aurik-verboten-linter`:
   `TONAL_NR_GATE_MISSING` — Aufruf einer NR-API (`_enhance_onnx`,
   `_enhance_torchscript`, `denoise(`, `process(` in den genannten
   Modulen) ohne `is_tonal_clean`-Referenz im selben Modul ⇒ ERROR.
2. **`[RELEASE_MUST]`-Test** `tests/normative/test_tonality_gate_contract.py`:
   - `test_flatness_reference_values`: feste Seeds ⇒ 0.00 / 0.69 / 0.99
     (Toleranz ±0.02).
   - `test_tonal_input_skips_ml_nr`: tonaler Eingang ⇒ ML-NR-Zweig nicht
     aufgerufen (Mock), DSP-Zweig läuft, Metadata gesetzt.
   - `test_noisy_input_reaches_ml_nr`: flatness 0.69 ⇒ ML-NR erreichbar.
3. **Konstanten-Guard**: `TONAL_CLEAN_FLATNESS` nur in
   `calibrated_constants.py` definiert (Static-Value-Guard).

## Entfernter Spielraum

- Wahl der Schwelle (fest 0.05, zentral verankert).
- Ort der Prüfung (Must-Call-Funktion + Lint).
- Duplizierung (ein kanonisches Modul statt Ad-hoc-Flatness-Checks,
  z. B. der heutige §2.47b-Block in phase_03 wird auf das Gate migriert).

## Betroffene Dateien

- NEU: `backend/core/dsp/tonality_gate.py`
- `backend/core/calibrated_constants.py` (Konstante)
- `backend/core/phases/phase_03_denoise.py`, `dsp/spectral_denoiser.py`,
  `backend/core/hybrid/hybrid_ml_denoiser.py` (Migration auf das Gate)
- `scripts/aurik_verboten_linter.py` (Regel + hartkodierter Katalog)
- `tests/normative/test_tonality_gate_contract.py`

## Risiken

- Zu aggressive Flatness-Schwelle könnte reale, tonarme Musik
  (Orgelpunkt, Drone) fälschlich als „tonal sauber" einstufen. Gegenmaßnahme:
  Referenzwert-Tests mit realen Korpus-Schnipseln vor Sign-off
  (Seed-Dokumentation im Evidenzblock).
