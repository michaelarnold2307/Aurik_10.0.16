# ID-Registry — Kanonische Identität aller §-Referenzen

> **Status: Aktiv — CI-enforced.** Kanonische Quelle für ID-Bedeutung und
> Zitierform. Pre-Commit-Hook `aurik-id-registry` prüft fail-closed
> (R1 unbekannte IDs, R2 nackte Ambiguitäts-Zitate). Drift-Schutz:
> `scripts/spec_drift_check.py` (WATCHED_FILES). Hintergrund und
> Bereinigungsplan: `docs/ID_COLLISION_MAP.md`. Zitierdisziplin:
> `AGENTS.md` §5. Validator: `scripts/id_registry_check.py` (Report-Modus
> und `--strict` für Pre-Commit, `--fix` für mechanische Qualifikation).

## Namensräume

Reihenfolge ist relevant: das erste passende Muster gewinnt. Backticks um
Muster sind optional — der Parser entfernt sie.

| Muster | Quelle | Status | Hinweis |
|---|---|---|---|
| `§G[1-9]` | `.github/copilot-instructions.md` | ambig | enforced (CI-geparst, hash-gewacht); Quelle immer angeben |
| `§V[1-9]` | `.github/copilot-instructions.md` | ambig | enforced; Quelle immer angeben |
| `§G[0-9]+` | `.github/GEBOTE.md` | enforced | Pre-Commit-Verifier (Teilmenge); §G1–§G9 ambig |
| `§SC-G[0-9]+` | `.github/GEBOTE.md` Startup-Block (jetzt Kategorie XXIV) | veraltet | Alias auf §G173–§G182 (Phase 1 umgesetzt) |
| `§G-DB[0-9]+` | `scripts/gebote_verifier.py` | verifier-intern | nicht in Code-Kommentaren zitieren |
| `G[0-9]{2}` | `.github/GEBOTEN.md` | Referenz | nicht in Code-Kommentaren zitieren |
| `§V[0-9]+` | gemischter §V10+-Raum | informell | GEBOTE VIII/XVI/XIX–XXIII (§V27–§V52), Spec-Vintage-Guards (§V19/§V24…), phase-lokale Release-MUST-Zitate; neue Zitate MÜSSEN Quelle angeben; §V19/§V24 sind ambig (Set) |
| `V[0-9]{2}` | `.github/VERBOTEN.md` | enforced | Linter V01–V52, fail-closed |
| `§v[0-9]+[A-Za-z0-9._-]*` | `.github/specs/` (versionierte Specs) | enforced | z. B. §v10.305, §v10.900 |
| `§[A-Z]{2,}[A-Z0-9-]*` | `.github/specs/`, `SPEC.md` | enforced | Subsysteme: §SFT, §UQ, §AC, §AF, §PID, §CSTC, §SLR-1, §CHT-1 |
| `§[0-9][A-Za-z0-9._-]*` | `.github/specs/`, `SPEC.md` | enforced | hierarchisch; `check_spec_refs.py` existiert, nicht verdrahtet |
| `§[GV]` | — | unspezifisch | bare ID ohne Nummer, präzisieren |
| `§[A-Z]` | altes A–Z-Pattern-Schema | veraltet | z. B. §H/§J/§U; auf GEBOTE-/SPEC-IDs migrieren |
| `§[A-Z][A-Za-z0-9._-]*` | freie Wort-Tags (Guard-/Modul-/Marker-Namen) | informell | kein normatives ID-System; präzisieren auf GEBOTE-/SPEC-ID |
| `§[a-z][A-Za-z0-9._-]*` | freie Wort-Tags (lowercase) | informell | kein normatives ID-System; präzisieren auf GEBOTE-/SPEC-ID |

## Ambiguitäts-Set

Nackte Zitate dieser IDs (ohne Quellen-Angabe in derselben Zeile) sind
VERBOTEN. Zitierform: **„§G4 (copilot-instructions.md)“** bzw.
**„§G4 (GEBOTE.md)“**.

| ID | Bedeutung (copilot-instructions.md) | Bedeutung (GEBOTE.md / andere Quelle) |
|---|---|---|
| §G1 | Song-Maximierung: pro Song isoliert, State-Reset | Pro-Song-Kalibrierung (global_scalar, Guards) |
| §G2 | Vollständige Defektbehebung (ganzer Song) | Defekt-Vollständigkeit: 62 DefectTypes pro Song |
| §G3 | Natürlicher Wohlklang | Gesangsintegrität (Vocal-Safety 80 Hz–8 kHz) |
| §G4 | CD-Rauschprofil-Pflicht (Export) | Ghost-Echo-Freiheit (§2.60 STCG) |
| §G5 | Deterministische Reproduzierbarkeit | Konsistenz-Mandat |
| §G6 | Psychoakustische Präzision | Null-Toleranz für Phasen-Leckage |
| §G7 | Chirurgische Defektbehandlung | Interchannel-Lag |
| §G8 | Transparenz (Audit-Log, Fallback-Logging) | CD-Rauschprofil-Pflicht (Export) |
| §G9 | Projektweite Konsistenz | Quellmaterial-Unabhängigkeit |
| §V1 | Vocal-Distortion-Verbot | VERBOTE.md §V1 (andere Bedeutung; veraltet) |
| §V2 | Ghost-Echo-Verbot | VERBOTE.md §V2 (andere Bedeutung; veraltet) |
| §V3 | Rauschprofil-Full-Song-Verbot | VERBOTE.md §V3 = Hard-Clamp (veraltet) |
| §V4 | Bridge-Bypass-Verbot | VERBOTE.md §V4 (andere Bedeutung; veraltet) |
| §V5 | Truncation-ohne-Dither-Verbot | VERBOTE.md §V5 (andere Bedeutung; veraltet) |
| §V6 | Silent-Failure-Verbot | VERBOTE.md §V6 (andere Bedeutung; veraltet) |
| §V7 | Workaround-Verbot | VERBOTE.md §V7 (andere Bedeutung; veraltet) |
| §V8 | Song-Cross-Contamination-Verbot | VERBOTE.md §V8 (andere Bedeutung; veraltet) |
| §V9 | Rauschprofil-Quellmaterial-Kopie-Verbot | VERBOTE.md §V9 (andere Bedeutung; veraltet) |
| §G71–§G75 | — (existiert dort nicht) | GEBOTE Kategorie IX: SFT-Adaptivität; Startup-Block hieß früher §SC-G71–§SC-G75 — Konflikt behoben (jetzt §G173–§G182, Kategorie XXIV) |
| §G76–§G80 | — (existiert dort nicht) | GEBOTE Kategorie X: Kalibrierungs-Dispatch; Startup-Block hieß früher §SC-G76–§SC-G80 — Konflikt behoben (jetzt §G173–§G182, Kategorie XXIV) |
| §G122–§G124 | — (existiert dort nicht) | Konflikt behoben: XI-b jetzt §G183–§G187; §G122–§G130 gehören eindeutig zu Kategorie XVIII |
| §V19 | — (existiert dort nicht) | Spec-Vintage-Guard „Noise-Texture-Detector“ vs. VERBOTE.md §V19 (veraltet, andere Bedeutung) |
| §V24 | — (existiert dort nicht) | Spec-Vintage-Guard „Tilt-Cap“ vs. VERBOTE.md §V24 (veraltet, andere Bedeutung) |

## Qualifikations-Mapping (für `--fix`)

Mechanische Quellen-Qualifikation für Bestandszitate des Ambiguitäts-Sets.
Basis: Stichproben-Audit der Code-Kontexte (2026-08). Neue Zitate werden
NICHT auto-fixiert — sie müssen von Hand korrekt qualifiziert werden
(Pre-Commit fail-closed).

| Muster | Qualifikator |
|---|---|
| `§G1`–`§G9` | (GEBOTE.md) |
| `§V1`–`§V9` | (copilot-instructions.md) |
| `§G71`–`§G80` | (GEBOTE.md) |
| `§G122`–`§G124` | (GEBOTE.md) |
| `§V19`, `§V24` | (Spec-Vintage-Guard) |

## Aliasse

| Alias-ID | Ziel (geplant) | Phase | Hinweis |
|---|---|---|---|
| §SC-G71–§SC-G80 | §G173–§G182 | 1 (umgesetzt) | Startup-Block GEBOTE.md, jetzt Kategorie XXIV; Alias-Vermerk je Zeile |
| §V1–§V35+ (VERBOTE.md) | — | — | kein Alias; veraltet — die normativen Codes sind V01–V52 (VERBOTEN.md) |
| G01–G36 (GEBOTEN.md) | — | — | eigener Referenz-Namensraum; nicht enforced |

## Pflege-Regeln

- **Neue ID einführen:** erst hier registrieren (Namensraum oder Einzelzeile),
  dann im Code zitieren — der Validator meldet nicht-registrierte IDs als
  WARNUNG (R1).
- **Ambiges Zitat:** ID ins Ambiguitäts-Set aufnehmen, wenn derselbe Bezeichner
  in zwei Quellen verschiedene Regeln meint.
- **Umbenennung:** alte ID in die Alias-Tabelle eintragen, Ziel-ID angeben —
  Zitate bleiben auflösbar, nichts hängt in der Luft.
- **Entfernen:** erst wenn kein Zitat im Repo mehr existiert
  (`scripts/id_registry_check.py` im Report-Modus zeigt Treffer).
