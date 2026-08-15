# Goldenes Hör-Set — Ohr-Messapparatur für Release-Entscheidungen

- Zweck: Der einzige Gate, der die Zielgröße selbst misst — **Unhörbarkeit der
  Defekte und Wohlklang für das menschliche Ohr**. Proxy-Metriken (PMGG,
  MERT-MUSHRA) bleiben advisory; dieser Gate ist das harte Ohr-Gate.
- Verwandte Dokumente: `docs/guides/GO_NO_GO_DECISION_PROTOCOL.md` (Prozess),
  `scripts/golden_set_tool.py`, `scripts/non_inferiority_gate.py`,
  `scripts/challenger_round.py`, `scripts/prepare_listening_study.py`.
- Status: Infrastruktur aktiv; Gate ist **fail-closed**, solange keine
  Hörurteile hinterlegt sind (`BLOCKED`).

## 1. Korpus (fix, versioniert, nie ersetzt)

- 56 Items, **fixer Bestand** — pro Release wird derselbe Corpus bewertet,
  nie neu gesampelt (sonst sind Verläufe nicht vergleichbar).
- Basis: der synthetische CC0-Corpus nach §15.2 (`corpus/README.md`,
  `corpus/*/manifest.yaml`, validiert durch `tests/corpus/test_corpus_integrity.py`).
  Jedes Item trägt eine **deklarierte Tonträgerkette** (`chain`-Feld im
  Material-Manifest) — die kuratierte Wahrheit; der MediumDetector liefert
  nur Querprüfungs-Evidenz (Mismatch wird gesperrt, nie still übernommen).
- Mehr-Generationen-Items (Depth 2–4+) werden deterministisch aus den
  Original-Quellen synthetisiert (`scripts/extend_corpus_chains.py`, fixer
  Seed, dokumentierte DSP-Proxy-Stufen, CC0). Echte historische Transfers
  ersetzen sie, sobald verfügbar.
- Coverage-Quoten (erzwungen durch `golden_set_tool.py check`):
  - Material: je ≥ 2 Items aus den 6 Transfer-Materialien des Korpus:
    vinyl, tape, shellac, digital, cassette, reel_tape.
  - Transfer-Chain-Tiefe: je ≥ 2 Items aus Depth 1, 2, 3, 4+.
- Manifest: `audit/golden_listening_set.json`
  (Schema: `items[{id, path, material, era_year, genre, defect_types,
  license, declared_chain, depth, restorability_score, …}]`).
  Erzeugung (nur degradierte Quellen — Zirkularitätsschutz):
  `python scripts/golden_set_tool.py init --corpus corpus --subdir damaged --classify`.
- **Kuration**: Detektor-Werte sind provisorisch (`detected_*`); authoritativ
  werden `depth`/`restorability_score` ausschließlich über
  `golden_set_tool.py verify` (Audit-Trail mit `verified_by`/`verified_at`;
  Material-Mismatch ohne deklarierte Kette wird verweigert).
  Unkurierte Items ⇒ Gate FAIL — nie still.

## 2. Hörurteile (Pflicht für jedes Gate-Urteil)

- Protokoll: MUSHRA nach ITU-R BS.1534-3 — Hidden Reference, 3.5-kHz-Anchor.
- Mindestens **10 Hörer pro Item**, blind, dokumentierte Abhörumgebung
  (Kopfhörer/kalibrierter Raum) — strenger als das GO/NO-GO-Minimum (N ≥ 8),
  damit pro Item ein Bootstrap-CI tragfähig ist.
- Verdict-Schema (JSON): siehe `scripts/non_inferiority_gate.py`.
- Hörer-Erschöpfung, fehlende Urteile oder unvollständige Abdeckung
  ⇒ `BLOCKED` (Exit 2), nie `PASS`.

## 3. Non-Inferiority-Entscheidung

- Pro Item: gepaarte Differenzen `candidate − anchor` pro Hörer;
  Percentile-Bootstrap (5 000 Ziehungen, fixer Seed ⇒ deterministisch, §G5).
- Item besteht: untere 95-%-CI-Grenze > −5.0 MUSHRA-Punkte (Marge).
- Gate besteht nur, wenn **alle** Items bestehen.
- Aufruf: `python scripts/non_inferiority_gate.py --verdicts <json>`
  (Exit 0 PASS / 1 FAIL / 2 BLOCKED).

## 4. Release-Regel

- Ein Release darf das goldene Set nicht signifikant verschlechtern.
- Die Entscheidung wird ausschließlich aus Hörurteilen abgeleitet; Proxy-
  Metriken dürfen sie nicht überstimmen und nicht ersetzen.
- Ergebnis wird versioniert abgelegt (Datei + Datum + Hörerzahl) und im
  Release-Bericht referenziert.

## 5. Challenger-Runden (Modellwechsel)

- Neuer Kandidat (z. B. AERO, Spec 04:225) tritt gegen den Incumbent auf dem
  goldenen Set an: `scripts/challenger_round.py prepare` baut das blinde
  Trial-Paket, `decide` wendet die Regel an:
  **ADOPT** nur bei (a) CI-Untergrenze `challenger − incumbent` > 0 **und**
  (b) bestandener Non-Inferiority gegen den Anchor.
- Ohne vollständige Urteile: `BLOCKED` — kein Modellwechsel auf Basis von
  Proxy-Metriken oder Annahmen.
