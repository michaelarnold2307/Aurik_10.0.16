# ID-Kollisions-Karte & Bereinigungsplan — Aurik 10

> **Status: nicht-normative Analyse** (Phase 0 des ID-Bereinigungsplans).
> Kanonische ID-Registry: `.github/ID_REGISTRY.md`. Zitierdisziplin:
> `AGENTS.md` §5. Validator (Report-Modus): `scripts/id_registry_check.py`.
>
> Stand der Zahlen: 92 Spec-Dateien, 818 Testdateien / 16.143 Testfunktionen,
> 426 §G-Zitate in backend/denker/forensics.

## 1. Kollisions-Karte (verifiziert)

| # | Kollision | Evidenz |
|---|---|---|
| K1 | `§G1`–`§G9` doppelt belegt: `copilot-instructions.md` vs. `GEBOTE.md` — verschiedene Regeln | §G4: copilot = CD-Rauschprofil-Pflicht, GEBOTE = Ghost-Echo-Freiheit |
| K2 | `§V1`–`§V9` doppelt belegt: `copilot-instructions.md` vs. `VERBOTE.md` (veraltet) | §V3: copilot = Rauschprofil-Full-Song-Verbot; VERBOTE.md:23 = Hard-Clamp |
| K3 | Doppelte Kategorie „XVIII“ in `GEBOTE.md` | Zeile 279 (Startup-Block) und Zeile 458 (§G122–§G130) → **behoben** (Startup-Block jetzt Kategorie XXIV) |
| K4 | Doppelte Kategorie „XXII“ in `GEBOTE.md` — identischer Titel „§G150–§G155 Metrik-Hierarchie & Guard-Disziplin“ | Zeile 298 und Zeile 547 → **behoben** (Duplikat entfernt, kanonisch bleibt der Block am Dateiende) |
| K5 | `§G122`–`§G124` doppelt beansprucht (Kategorie XI-b und XVIII) | Zeile 249 und Zeile 458 → **behoben** (XI-b jetzt §G183–§G187) |
| K6 | Startup-Regeln notdürftig zu `§SC-G71`–`§SC-G80` umbenannt; `§G71`–`§G80` sind bereits von Kategorie IX (§G68–§G75) und X (§G76–§G81) belegt | Zeilen 213, 226, 279–296 → **behoben** (Startup jetzt §G173–§G182) |
| K7 | `CLAUDE.md` zitiert die Startup-Regeln als §G71–§G74 → diese Zitate zeigen falsch auf Kategorie IX | 14 §G71-Zitate im Code → **behoben** (CLAUDE.md zitiert jetzt §G173–§G176) |
| K8 | Verifier-interner Sub-Namensraum `§G-DB1`–`§G-DB6` ohne Katalog-Eintrag | `scripts/gebote_verifier.py` |
| K9 | Zwei weitere, gebannerte Namensräume: `G01`–`G36` (GEBOTEN.md) und `§V1`–`§V35`+ (VERBOTE.md) | beide Dateien tragen Status-Banner |
| K10 | Physische Reihenfolge teils defekt: XXIII stand vor XV; zwei Änderungshistorie-Tabellen | `GEBOTE.md` → **behoben** (XXIII hinter XXII einsortiert, Historien zu einer Tabelle zusammengeführt) |
| K11 | Spec-Vintage-Guards (`§V19`, `§V24`, `§V38`, `§V40`, `§V41`) vs. `VERBOTE.md`-§V-Bereich — kollidierende Bezeichner; vom Validator als ambig gemeldet (73× §V24) | `.agents/skills/spec/SKILL.md`, `VERBOTE.md` |
| K12 | Undokumentierter Namensraum `§Perf` im Code (36 Zitate) — vom Validator als R1 entdeckt | `scripts/id_registry_check.py` Report |

## 2. Namensraum-Übersicht

| Namensraum | Datei | Status |
|---|---|---|
| §G1–§G9, §V1–§V9 | `.github/copilot-instructions.md` | enforced (CI-geparst, hash-gewacht) |
| §G1–§G187 | `.github/GEBOTE.md` | enforced (Pre-Commit-Verifier, hartkodierte Teilmenge) |
| §SC-G71–§SC-G80 | `.github/GEBOTE.md` (Startup-Block) | veraltet — Alias auf §G173–§G182 (Kategorie XXIV) |
| §G-DB1–§G-DB6 | `scripts/gebote_verifier.py` | verifier-intern, nicht zitieren |
| G01–G36 | `.github/GEBOTEN.md` | Referenz (nicht enforced) |
| §V1–§V35+ | `.github/VERBOTE.md` | veraltet (Banner) |
| V01–V52 | `.github/VERBOTEN.md` | enforced (Linter, fail-closed) |
| §0–§9, §2.xx, §v10.xxx | `.github/specs/`, `SPEC.md` | hierarchisch; `scripts/compliance/check_spec_refs.py` existiert, ist aber nicht verdrahtet |

## 3. Bereinigungsplan (5 Phasen)

- **Phase 0 (erledigt):** Kollisions-Karte fixieren — dieses Dokument.
- **Phase 1 (erledigt):** `GEBOTE.md` bereinigt — Details und Ergebnis in §4.
- **Phase 2 (erledigt):** Ambiguitäts-Set maschinell disambiguiert —
  498 nackte Zitate in 166 Dateien mechanisch qualifiziert
  (`scripts/id_registry_check.py --fix`, Qualifikations-Mapping in der
  Registry, Basis: Stichproben-Audit der Code-Kontexte); der §V10+-Raum wurde
  als „informell“ klassifiziert.
- **Phase 3 (erledigt):** Pre-Commit-Hook `aurik-id-registry` fail-closed
  verdrahtet (R1 unbekannte IDs, R2 nackte Ambiguitäten). Endzustand:
  0 WARNUNGEN repo-weit, `--strict` Exit 0, Hook --all-files Passed.
- **Phase 4 (erledigt):** R1-Einzelgänger (freie Wort-Tags wie §Perf,
  §Frisson, §Wall-Time-Budget) über Catch-all-Namensräume in der Registry
  registriert — Grandfathering ohne Informationsverlust; neue normative IDs
  bleiben fail-closed.
- **Phase 5 (erledigt):** `.github/ID_REGISTRY.md`, `docs/ID_COLLISION_MAP.md`
  und `AGENTS.md` in `scripts/spec_drift_check.py` `WATCHED_FILES`
  aufgenommen — das ID-System ist drift-geschützt.

## 4. Phase 1 — umgesetzt (GEBOTE.md bereinigt)

Angewendet, semantik-neutral, Aliasse erhalten:

1. **XXII-Duplikat entfernt:** Der frühere Block war ein Duplikat des
   kanonischen Blocks am Dateiende — entfernt.
2. **Startup-Block umgehängt:** `§SC-G71`–`§SC-G80` → **§G173–§G182** unter
   neuer Kategorie **XXIV — Startup-Integration & Kommunikation**;
   Alias-Vermerk „früher §SC-G7x“ je Zeile. Hinweis: `§G167`–`§G172` waren
   bereits durch „Denker-IQ & Material-Awareness“ belegt — daher §G173 ff.
3. **XI-b re-ID’t:** `§G122`–`§G126` → **§G183–§G187**; Kategorie XVIII
   behält §G122–§G130. Header korrigiert.
4. **XIX-Header:** Bereich auf §G131–§G137 korrigiert (§G137 existierte
   bereits als B3-P2-Regel).
5. **CLAUDE.md:** Startup-Zitate auf §G173–§G176 umgestellt; Plugin-Kommentare
   (§SC-G72 → §G174) in 5 Plugin-Dateien nachgezogen.
6. **Historie:** Eintrag 10.0.14 (2026-07-30) auf §G173–§G182 / Kategorie XXIV
   korrigiert.
7. **Reihenfolge bereinigt (K10):** XXIII hinter XXII einsortiert, beide
   Änderungshistorie-Tabellen zu einer zusammengeführt, neuer
   Historie-Eintrag 10.0.19 ergänzt. Der Pre-Commit-Verifier prüft die neuen
   Bereiche nicht → keine Gate-Wirkung. `copilot-instructions.md` unangetastet.

## 5. Warum kein Verlust von Features, Qualität oder Stabilität

- **Kein Laufzeitcode:** Es werden nur Dokumente, Kommentarzeilen und ein
  additives Skript geändert. UV3, DSP, Dither, `global_scalar`-Logik und
  Determinismus (§G5) bleiben unberührt → bit-identische Ausgaben.
- **Keine Verifier-Semantik-Änderung:** `gebote_verifier.py` prüft weiter
  dieselben hartkodierten IDs; `test_spec_consistency.py`, Release-Must-Parser
  und Verbots-Linter laufen unverändert; die 16.143 Tests sind nicht betroffen.
- **Additive Enforcement-Only:** Der neue Check blockiert nur **neue**
  Verstöße; die Bestandszitate sind per Grandfathering immun.
- **Rückrollbar:** Jede Phase ist einzeln revertierbar (Dokument, Skript,
  Kommentar-Diff).

**Bewusst verworfen:** globales UIN-System (§AUR-G-0001) — sauber, aber
destruktiv: Churn über 426 Zitate + Verifier-Hardcodes ohne funktionalen
Gewinn; eine Umbenennung der §G1–§G9 in `copilot-instructions.md` würde
zusätzlich die CI-Hash-Kette brechen.
