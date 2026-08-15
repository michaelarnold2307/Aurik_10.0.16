# Vorschlag 05 — Pipeline-Invariante: Kein erkanntes Hörbares unbehandelt

> **Status: VORSCHLAG (Entwurf) — nicht normativ.** Übernahme als
> Ergänzung der Spec `02_pipeline_architecture.md` + Umsetzung in
> `backend/core/unified_restorer_v3.py`.
> Formel und Meta-Regel: `00_interpretationsfrei_uebersicht.md`.

## Zweck

Mess-Evidenz aus der Session (Backend-Log 14.08.): 32 Phasen geplant →
21 geprunt (§AC) → 10 weitere Skip-Gates → **genau 1 Phase lief**,
obwohl der Scanner hum=1.00, flutter=1.00, bandwidth_loss=1.00 meldete.
Die Skip-Kette kann das Kernprinzip „Niemals einen erkannten Defekt
unbehandelt lassen" (README) kollektiv aushebeln. Es fehlt eine
End-zu-End-Invariante mit maschinellem Nachweis.

## Normativer Wortlaut (für Spec 02-Ergänzung)

> `[RELEASE_MUST] §2-DEFEKT-VOLLSTÄNDIGKEIT: Für jeden Defekt mit
> severity ≥ `AUDIBILITY_FLOOR` (CalibratedConstants, aktuell 0.03) und
> Eintrag in `CAUSE_TO_PHASES` MUSS nach der Execution gelten:
>
> (a) mindestens eine zugeordnete Phase wurde ausgeführt, ODER
> (b) ein Skip-Grund wurde protokolliert, der exakt eines der
> zulässigen Muster trägt: `§v10.707` (Defekt-Absenz — nur zulässig,
> wenn die Severity zum Ausführungszeitpunkt nachweislich < Floor),
> `§v10.24` (bereits behoben — nur mit Referenz auf die Phase, die
> behoben hat), `§v10.303` (Low-Confidence — nur Enhancement-Familien),
> `§2.70` (Kalibration lehnt ab).
>
> `RestorationResult.metadata` MUSS die Liste `unresolved_defects`
> enthalten (leer bei vollständiger Abdeckung). Bei Processing Intensity
> = 0.0 % und nicht-leerer Defektliste MUSS `logger.warning` mit
> §-Referenz erfolgen — kein stilles Passthrough-Ergebnis.

## Enforcement

1. **Integrationstest** `tests/integration/test_defect_completeness_invariant.py`:
   - Mock-Scanner injiziert {hum: 1.0, flutter: 1.0, bandwidth_loss: 1.0}
     und erzwungene Skips; prüft (a) `unresolved_defects`-Vertrag,
     (b) Warning-Vertrag, (c) dass Skips nur mit zulässigen Mustern
     erfolgen.
2. **Log-Kontrakt-Checker** `scripts/check_pipeline_log_contract.py`
   (nightly): parst das Backend-Log auf „Niemals-unbehandelt"-Verstöße
   (Defekt gemeldet, keine Phase, kein zulässiger Skip-Grund).
3. UV3-intern: `_should_skip_*`-Gate-Reihenfolge und -Begründungen werden
   an zentraler Stelle (ein Dispatch) gesammelt statt verstreut — per
   AST-Test `test_skip_gates_centralized` geprüft.

## Entfernter Spielraum

- Ob ein Defekt „abgehakt" ist (vollständige Kette a/b mit zulässigen
  Mustern statt freier Skip-Gründe).
- Stille Passthroughs (Warning-Pflicht + Log-Checker).
- Verstreute Gate-Logik (zentraler Dispatch).

## Betroffene Dateien

- `backend/core/unified_restorer_v3.py` (Invariante + `unresolved_defects`)
- `backend/core/calibrated_constants.py` (AUDIBILITY_FLOOR zentral)
- `tests/integration/test_defect_completeness_invariant.py` (NEU)
- `scripts/check_pipeline_log_contract.py` (NEU, nightly)
- `tests/unit/test_unified_restorer_v3.py` (AST-Test Skip-Dispatch)

## Risiken

- Falsch-positive Scanner-Meldungen auf Kurzclips könnten die Invariante
  mit Lärm füllen — Gegenmaßnahme: der Floor + ein Mindest-Signaldauer-
  Gate im Scanner (≥ 2 s) werden in die Invariante aufgenommen.
