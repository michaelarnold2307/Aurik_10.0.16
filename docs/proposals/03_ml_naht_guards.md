# Vorschlag 03 — ML-Naht-Guards: Polarity & Lag in hybrid_ml_apply

> **Status: VORSCHLAG (Entwurf) — nicht normativ.** Übernahme als
> `VERBOTEN.md` V53 + Erweiterung von `backend/core/dsp/hybrid_ml_blend.py`.
> Formel und Meta-Regel: `00_interpretationsfrei_uebersicht.md`.

## Zweck

Mess-Evidenz aus der Session: Der SGMSE+-TorchScript-Pfad lieferte
**polaritätsinvertierte** Ausgabe (Korrelation −0.85; Ursache:
Score-Negation im Eager-Zweig / Konventions-Abweichung des Checkpoints).
Jede ML-Bridge ist ohne Guards ein eigenes Natürlichkeits-Risiko.

## Normativer Wortlaut (für VERBOTEN.md V53)

> `[RELEASE_MUST] §V53 ML-NAHT-GUARDS: `backend.core.dsp.hybrid_ml_blend.
> hybrid_ml_apply` MUSS vor dem Blend deterministisch prüfen:
>
> 1. **Polarity-Guard**: Pearson-Korrelation von dry und wet (zentriert)
>    < 0.0 ⇒ Ausgabe gilt als invertiert; Vorzeichen wird korrigiert und
>    `metadata["polarity_corrected"] = True` protokolliert.
> 2. **Lag-Guard**: Kreuzkorrelationsmaximum mit |lag| >
>    `MAX_ML_LAG_SAMPLES` ⇒ ML-Ausgabe wird verworfen (dry zurück),
>    `metadata["lag_rejected"] = True`.
>
> `MAX_ML_LAG_SAMPLES = 128` liegt in `CalibratedConstants`.
> Direkte Blend-Ausdrücke (`dry + w * (wet - dry)`) außerhalb von
> `hybrid_ml_apply` sind in `backend/core/`, `plugins/` und `dsp/`
> VERBOTEN. Jede ML-Bridge (SGMSE+, DFN, MIIPHER, hybrid_ml_denoiser,
> harmonic_inpainting, whisper_denoiser) MUSS ihren Wet-Ausgang über
> `hybrid_ml_apply` führen; ein Plugin-eigener skalarer Fallback ist nur
> zulässig, wenn `backend` nicht importierbar ist (Standalone-Betrieb)
> und muss dann beide Guards äquivalent implementieren.

## Enforcement

1. `[RELEASE_MUST]`-Tests `tests/unit/test_ml_naht_guards.py`:
   - `test_inverted_wet_is_corrected`: wet = −dry ⇒ Ausgabe = dry,
     Flag gesetzt.
   - `test_shifted_wet_is_rejected`: wet = np.roll(dry, 256) ⇒ dry zurück,
     Flag gesetzt.
   - `test_plugin_bridges_use_hybrid_ml_apply` (AST-Test): die sechs
     Bridges enthalten keinen eigenen Blend-Ausdruck im Normalbetrieb.
2. Lint (fail-closed): Blend-Muster-Regex in Produktionscode ⇒ ERROR.

## Entfernter Spielraum

- Wo Guards leben (exakt ein Ort) und wie streng (Schwellen fest).
- Plugin-eigene Naht-Implementierungen (verboten bzw. nur mit
  äquivalenten Guards im Standalone-Fallback).

## Betroffene Dateien

- `backend/core/dsp/hybrid_ml_blend.py` (Guards + Metadata)
- `backend/core/calibrated_constants.py` (Konstante)
- `plugins/sgmse_plugin.py`, `plugins/deepfilternet_v3_ii_plugin.py`,
  `plugins/miipher_plugin.py`, `backend/core/hybrid/hybrid_ml_denoiser.py`,
  `plugins/harmonic_inpainting_plugin.py`,
  `plugins/whisper_denoiser_plugin.py` (Aufruf-Migration)
- `scripts/aurik_verboten_linter.py` (Blend-Muster-Regel)
- `tests/unit/test_ml_naht_guards.py` (NEU)

## Risiken

- Polarity-Guard auf bewusst phaseninvertierten Signalen (z. B. invertierte
  Aufnahme) wäre ein Fehlgriff — Gegenmaßnahme: Guard nur auf
  ML-Enhancement-Naht, nicht auf generisches Audio; dokumentiert im
  Evidenzblock mit Korpus-Fällen.
