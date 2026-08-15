# Vorschlag 00 — Interpretationsfreie Regeln: Meta-Regel & Übersicht

> **Status: VORSCHLAG (Entwurf) — nicht normativ.**
> Dieses Dokument liegt bewusst in `docs/proposals/` und NICHT unter
> `.github/specs/`. Es wird erst nach Maintainer-Sign-off in die normative
> Kette überführt (dann mit den üblichen PR-Pflichten: Evidenzblock, Seed,
> 95 %-CI, Maintainer Sign-off gemäß `AGENTS.md` §4).
> Arbeitssprache: Deutsch.

## Zweck

Interpretationsspielraum entsteht dort, wo eine Regel keinen maschinellen
Prüfpunkt hat. Diese Übersicht definiert die **Formel**, nach der alle
nachfolgenden Vorschläge (01–08) formuliert sind, und die **Meta-Regel 0**,
die erzwingt, dass künftige normative Regeln nie wieder ohne Prüfpunkt
entstehen.

## Meta-Regel 0 — Maschinen-Prüfpunkt-Pflicht

**Normativer Wortlaut (für spätere Übernahme in `.github/copilot-instructions.md`):**

> `[RELEASE_MUST] §G-REGELN-MASCHINE: Jede normative Regel in der Kette
> (copilot-instructions, VERBOTEN, instructions/, specs/) MUSS mindestens
> einen maschinellen Prüfpunkt besitzen — einen `[RELEASE_MUST]`-Test,
> eine fail-closed-Lint-/Verifier-Regel oder ein CI-Gate — ODER im selben
> Absatz explizit als `advisory` markiert sein. Der Coverage-Check
> `scripts/rule_machine_coverage_check.py` (Teil von `ci-lite.yml`) gleicht
> jeden §-Header der normativen Dokumente gegen die hartkodierten
> Verifier-Regeln und die Test-Suite ab und schlägt fehl (fail-closed),
> wenn ein normativer Absatz ohne Prüfpunkt und ohne `advisory`-Marker
> existiert.

**Warum das Spielraum eliminiert:**
Die heute dokumentierte Pflicht „Regeländerung ⇒ Skript nachziehen"
(AGENTS.md §2) wird selbst maschinell überwacht. Disziplin wird durch
Nachweis ersetzt.

## Die Formel (gilt für alle Vorschläge 01–08)

Jeder Vorschlag beantwortet exakt vier Fragen maschinell prüfbar:

1. **Was ist die Schwelle?** — quantifizierter Wert, verankert in
   `backend/core/calibrated_constants.py` (eine zentrale Konstantenquelle,
   keine verstreuten Magic Numbers).
2. **Wo gilt es?** — erschöpfende Modul-/Aufruf-Liste ODER kanonische
   Must-Call-Funktion.
3. **Was passiert bei Verstoß?** — fail-closed-Lint/Verifier/Gate mit
   definierter Fehlermeldung; kein „sollte/möglichst".
4. **Wer beweist es?** — `[RELEASE_MUST]`-Test, dessen Existenz
   `scripts/release_must_coverage_check.py` erzwingt.

## Zuordnung: Hebel → Vorschlag → zukünftiger normativer Ort

| # | Hebel | Vorschlag | Zukünftiger normativer Ort |
|---|---|---|---|
| 1 | Tonalität wird für Rauschen gehalten | `01_tonality_aware_nr.md` | neue Spec `23_tonality_aware_nr.md` + `CalibratedConstants` |
| 2 | Metriken widersprechen sich | `02_metrik_konsistenz.md` | Spec `v10.703_*`-Ergänzung + `mert_mushra_proxy.py` |
| 3 | ML/DSP-Naht ohne Konventions-Guards | `03_ml_naht_guards.md` | `VERBOTEN.md` V53 + `hybrid_ml_blend.py` |
| 4 | STFT-Konventions-Abweichler | `04_stft_konvention.md` | `VERBOTEN.md` V54 + `aurik-verboten-linter` |
| 5 | Gate-Akkumulation ohne Invariante | `05_pipeline_invariante.md` | Spec 02-Ergänzung + `unified_restorer_v3.py` |
| 6 | Statische Gates laufen ins Leere | `06_static_gates.md` | copilot-instructions §Performance/CI + `.pre-commit-config.yaml` |
| 7 | Verifikationsschuld | `07_verifikationsartefakt.md` | `solo-release-gate.yml` |
| 8 | Parallele Agenten ohne Vertrag | `08_koordinationsvertrag.md` | `AGENTS.md` §9 + PR-Template-Checkliste |

## Umsetzungsreihenfolge (Vorschlag)

1. **Meta-Regel 0** zuerst: `rule_machine_coverage_check.py` schreiben und in
   `ci-lite.yml` verdrahten (Report-Modus → fail-closed nach Stabilisierung).
2. **Hebel 1 + 2 + 3** (schützen und kalibrieren direkt die
   Klangwahrnehmung).
3. **Hebel 5 + 6** (verhindern Untätigkeit und stilles Kippen).
4. **Hebel 4, 7, 8** (Hygiene).

## Bewusst NICHT Teil dieses Vorschlags

- Die Evidenz-Gates (Korpus ≥ 20 Real-Audio-Cases, HPI ≥ 0.60, Quality ≥ 0.70,
  Commitment C2) bleiben ein **Datenprojekt** (Korpus-Generierung,
  Hörpanel) und werden hier nur in Vorschlag 02 metrisch flankiert.
- Änderungen an Algorithmus-Wahl (z. B. OMLSA vs. Wiener) — dort bleibt
  genuin kreativer Entscheidungsspielraum, der per §4-Evidenzblock
  dokumentiert wird.

## Übernahme-Pflichten (bei Maintainer-Sign-off)

- Alle in 01–08 neu eingeführten §-IDs (`§V53`, `§V54`, `§23-TONALITY`,
  `§MUSHRA-LUFS`, `§VERDICT-PASSTHROUGH`, `§METRIK-KALIBRATION`,
  `§2-DEFEKT-VOLLSTÄNDIGKEIT`, `§CI-STATIC-GATES`,
  `§CI-FULLSUITE-ARTEFAKT`) MÜSSEN in `.github/ID_REGISTRY.md`
  registriert werden (Namensraum `V[0-9]{2}` = VERBOTEN.md, enforced;
  übrige mit Quellen-Angabe).
- Neue Konstanten werden ausschließlich in
  `backend/core/calibrated_constants.py` angelegt (Static-Value-Guard
  prüft das).
- Jeder übernommene Vorschlag bekommt seinen `[RELEASE_MUST]`-Test,
  dessen Existenz `scripts/release_must_coverage_check.py` erzwingt —
  und die Meta-Regel 0 erzwingt beides gemeinsam.
