# Vorschlag 02 — Metrik-Konsistenz: MUSHRA-Proxy & Verdikt-Semantik

> **Status: VORSCHLAG (Entwurf) — nicht normativ.** Übernahme als
> Ergänzung der Spec `v10.703_perzeptueller_autopilot.md` bzw. in
> `backend/core/mert_mushra_proxy.py` + `musical_quality_assurance.py`.
> Formel und Meta-Regel: `00_interpretationsfrei_uebersicht.md`.

## Zweck

Mess-Evidenz aus der Session: PQS bewertete einen Eingang als „excellent
(83.6)", der MUSHRA-Proxy als „Poor (43)" — bei LUFS-Δ von **−29.3 LU**
gegen die Referenz und einem Verdikt, das bei 0 % Verarbeitung
(„43→43") eine Regression auswies. Solange Metriken einander widersprechen,
kann die Entscheidungsschicht nicht fürs menschliche Ohr optimieren.

## Normativer Wortlaut (für Spec-Ergänzung)

> `[RELEASE_MUST] §MUSHRA-LUFS: Der MUSHRA-Proxy MUSS beide Eingänge
> (Referenz und Kandidat) VOR jeder Feature-Berechnung auf −23 LUFS
> (ITU-R BS.1770-4) normalisieren. Ein Vergleich mit |LUFS-Δ| > 1.0 LU
> nach Normalisierung ist VERBOTEN; in diesem Fall MUSS das Ergebnis als
> `invalid_loudness` markiert werden (Score wird nicht ausgewiesen).
> Die Ziel-Lautheit `MUSHRA_TARGET_LUFS = -23.0` liegt in
> `CalibratedConstants`.
>
> `[RELEASE_MUST] §VERDICT-PASSTHROUGH: Weist `musical_quality_assurance`
> eine Processing Intensity < 5 % aus (Wert aus `modules_applied`),
> MUSS der Verdikt `no_processing_applied` gesetzt werden und
> `musical_improvement` exakt 0.0 betragen. Ein Qualitäts-Verdikt
> (PASS/FAIL) auf Passthrough ist VERBOTEN — die Bewertung beschreibt
> sonst den Eingang, nicht die Verarbeitung.
>
> `[RELEASE_MUST] §METRIK-KALIBRATION: Die Kalibrierungs-Suite
> `tests/musical_goals/test_metric_agreement.py` MUSS für jedes Release
> laufen: auf einem festen Gut/Schlecht-Paar-Set (Seed dokumentiert)
> müssen PQS- und MUSHRA-Rangfolge in ≥ 95 % der Fälle übereinstimmen.
> Abweichungen sind Release-blockierend.

## Enforcement

1. `[RELEASE_MUST]`-Tests:
   - `test_proxy_lufs_invariant`: Paar mit ±20-LU-Offset liefert nach
     Normalisierung denselben Score ±0.01.
   - `test_verdict_passthrough`: 0 Module ⇒ `no_processing_applied`,
     improvement 0.0.
   - `test_metric_agreement`: Gut/Schlecht-Rangfolge ≥ 95 %.
2. Lint: `musical_improvement`-Zuweisung ohne Intensity-Check ⇒ ERROR
   (fail-closed).

## Entfernter Spielraum

- Ob normalisiert wird (Pflicht) und auf welches Ziel (−23 LUFS, fest).
- Verdikt-Semantik bei Passthrough (fest definiert).
- „Metriken sollten zusammenpassen" ⇒ maschineller Agreement-Test.

## Betroffene Dateien

- `backend/core/mert_mushra_proxy.py` (LUFS-Normalisierung)
- `backend/core/musical_quality_assurance.py` (Verdikt-Semantik)
- `backend/core/calibrated_constants.py` (Konstante)
- `tests/musical_goals/test_metric_agreement.py` (NEU)
- `scripts/release_must_coverage_check.py` (deckt die neuen Header ab)

## Risiken

- LUFS-Normalisierung verändert bestehende Proxy-Scores historisch —
  im Evidenzblock als bewusste Re-Kalibrierung ausweisen (Seed + Vorher/
  Nachher-Werte).
