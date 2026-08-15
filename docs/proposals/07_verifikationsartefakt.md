# Vorschlag 07 — Verifikationsartefakt: Merge nur mit belegtem Grün

> **Status: VORSCHLAG (Entwurf) — nicht normativ.** Übernahme in
> `.github/workflows/solo-release-gate.yml` + `nightly-quality.yml`.
> Formel und Meta-Regel: `00_interpretationsfrei_uebersicht.md`.

## Zweck

Mess-Evidenz aus der Session: Testsuite wuchs von ~5.1k auf ~19.7k Tests;
der letzte dokumentierte grüne Gesamtlauf war von **April**; der Cache
des 13.08. enthielt 2.128 Fehlschläge, die sich als fast vollständig
transient herausstellten — aber es existierte kein Artefakt, das dies
belegte. „Vermutlich grün" ist kein Zustand.

## Normativer Wortlaut (für solo-release-gate)

> `[RELEASE_MUST] §CI-FULLSUITE-ARTEFAKT: Ein Merge auf `main` setzt ein
> Vollsuite-Artefakt voraus, das höchstens **7 Kalendertage** alt ist und
> von der Nightly-Pipeline erzeugt wurde:
>
> - `logs/fullsuite_latest.log` — vollständiges pytest-Protokoll
>   (Exit-Code dokumentiert), und
> - `reports/fullsuite_summary.md` — maschinell generierter Kurzreport
>   (Datum, HEAD-SHA, Tests gesamt/passed/failed/skipped/deselected,
>   Fehlerliste, Laufzeit).
>
> Das Gate-Skript `scripts/check_fullsuite_artifact.py` MUSS
> (a) Dateialter ≤ 7 Tage und (b) Exit-Code 0 im Artefakt prüfen; bei
> Verletzung wird der Merge blockiert. Lokale Teil-Läufe ersetzen das
> Artefakt NICHT.

## Enforcement

1. `scripts/check_fullsuite_artifact.py` (NEU), verdrahtet in
   `solo-release-gate.yml`.
2. Nightly-Workflow: Vollsuite-Lauf + Artefakt-Erzeugung (die Artefakte
   sind gitignored, liegen aber auf dem CI-Runner bzw. werden als
   Workflow-Artifact publiziert).
3. Selbsttest: Skript erzeugt künstlich altes Artefakt ⇒ Blockade,
   frisches grünes ⇒ Freigabe.

## Entfernter Spielraum

- „Suite ist grün" ⇒ belegtes Grün mit Ablaufdatum.
- Welche Läufe zählen (nur Nightly-Artefakt).

## Betroffene Dateien

- `.github/workflows/solo-release-gate.yml`, `nightly-quality.yml`
- `scripts/check_fullsuite_artifact.py` (NEU)
- `.gitignore` (logs/fullsuite_latest.log, reports/fullsuite_summary.md)

## Risiken

- 7-Tage-Fenster kann bei langer Feature-Pause unnötig blocken —
  Gegenmaßnahme: Nightly erzeugt das Artefakt automatisch; das Fenster
  ist eine Konstante im Skript und per Maintainer-Sign-off änderbar.
