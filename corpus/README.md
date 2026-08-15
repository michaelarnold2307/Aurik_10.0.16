# Aurik Echt-Audio-Corpus — §15.2

## Zweck

Dieses Verzeichnis enthält echte Musikaufnahmen für reproduzierbare Qualitätsmessung.
Auriks 18.400+ Tests operieren auf synthetischen Signalen — dieser Corpus schließt die
Lücke zwischen synthetischer Validierung und echtem Höreindruck.

## Verzeichnisstruktur

```
corpus/
├── MANIFEST_SCHEMA.yaml          # JSON-Schema für manifest.yaml
├── README.md                     # Diese Datei
├── shellac/
│   ├── manifest.yaml
│   ├── damaged/                  # Defekte Original-Aufnahmen
│   └── restored/                 # Aurik-restaurierte Versionen
├── vinyl/
│   ├── manifest.yaml
│   ├── damaged/
│   └── restored/
├── tape/
│   ├── manifest.yaml
│   ├── damaged/
│   └── restored/
├── reel_tape/
│   ├── manifest.yaml
│   ├── damaged/
│   └── restored/
├── cassette/
│   ├── manifest.yaml
│   ├── damaged/
│   └── restored/
└── digital/
    ├── manifest.yaml
    ├── damaged/
    └── restored/
```

## Rechtlicher Hinweis

**Alle Dateien in diesem Corpus MÜSSEN entweder gemeinfrei (Public Domain) oder unter
einer CC0-Lizenz stehen.** Urheberrechtlich geschütztes Material ist EXPLIZIT
AUSGESCHLOSSEN. Jeder Eintrag in einer `manifest.yaml` MUSS das Feld `license` führen.

Füge KEINE eigenen MP3s, FLACs oder WAVs hinzu, deren Lizenzstatus unklar ist.
Nutze stattdessen `scripts/generate_corpus_from_public_domain.py`, um automatisch
Public-Domain-Material von vertrauenswürdigen Quellen herunterzuladen.

## Quellen für Public-Domain-Aufnahmen

| Quelle | URL | Material |
|--------|-----|----------|
| Internet Archive | <https://archive.org/details/78rpm> | Shellac, Vinyl |
| Musopen | <https://musopen.org> | Klassik (Shellac, Vinyl, Tape) |
| Freesound (CC0) | <https://freesound.org> | Einzelklänge, Atmosphären |
| Library of Congress | <https://loc.gov/audio/> | Historische Aufnahmen |
| Europeana Sounds | <https://www.europeana.eu> | Europäisches Audio-Erbe |

## Bereits integrierte Public-Domain-Aufnahmen (verifiziert 2026-08-15)

`corpus/shellac/damaged/` enthält 12 echte Public-Domain-Aufnahmen aus der
Internet Archive 78rpm Collection (1905–1925, US-Public-Domain, Public Domain
Mark 1.0). Alle Einträge tragen `source_url` (archive.org/details/…),
`source_attribution` und eine verifizierte `checksum_sha256`
(`test_corpus_integrity` grün). Die downloadbaren URLs liegen in
`scripts/generate_corpus_from_public_domain.py`.

Die übrigen Material-Kategorien (Vinyl, Tape, Kassette, Digital) enthalten
selbst generierte CC0-Referenzen (`scripts/generate_corpus.py`), da für diese
Formate (nach 1930) faktisch keine gemeinfreien Aufnahmen existieren.

## Manifest

Jedes Unterverzeichnis führt eine `manifest.yaml` nach dem Schema in
`MANIFEST_SCHEMA.yaml`. Validierung:

```bash
python tests/corpus/test_corpus_integrity.py
```

## Pipeline Smoke Test

```bash
python -m pytest tests/corpus/test_corpus_pipeline_smoke.py -v
```

## Mindestanforderungen (Quality Gate)

- ≥ 20 Aufnahmen in ≥ 4 Material-Kategorien
- ≥ 5 Vokal-Aufnahmen
- Alle Manifest-Einträge valide (test_corpus_integrity grün)
- Kein Crash in Pipeline-Smoke-Test (test_corpus_pipeline_smoke grün)

## Aufnahmen hinzufügen

1. Audiodatei im passenden `damaged/`-Ordner ablegen (FLAC oder WAV, 48 kHz empfohlen)
2. Eintrag in `manifest.yaml` mit `file`, `duration_s`, `sample_rate`, `material`, `era_year`, `genre`, `license`, `source_attribution` ausfüllen
3. Optional: `source_url`, `defect_types`, `vocal`, `language`
4. `test_corpus_integrity` laufen lassen
5. Commit mit aussagekräftiger Message (z.B. "corpus: add 1950s jazz shellac recording")
