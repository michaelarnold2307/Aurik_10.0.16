# Community-Governance — Aurik 10

## Projektstatus

Aurik 10 ist ein Open-Source-Desktop-Projekt (Apache-2.0).
Der Hauptentwickler ist Michael Arnold. Externe Beiträge sind herzlich willkommen.

---

## Entscheidungsstruktur

### Hauptentwickler (Benevolent Dictator)

Michael Arnold entscheidet über:

- Architekturentscheidungen (Pipeline, Musical Goals, API-Stabilität)
- Release-Zeitplan und Versioning
- Akzeptanz oder Ablehnung von Pull Requests mit Architektur-Konsequenzen
- Modell-Auswahl und SOTA-Upgrades

### Community-Beitragende

Beiträge werden nach folgenden Kriterien bewertet:

1. **Korrektheit** — Alle bestehenden Tests grün, Musical Goals nicht verletzt
2. **Spec-Konformität** — Einhaltung von copilot-instructions.md
3. **Out-of-the-Box-Pflicht** — Keine neuen Netzwerkabhängigkeiten zur Laufzeit
4. **Qualität** — ruff + black + mypy strict bestanden

### RFC-Prozess (Request for Comments)

Für größere Änderungen (neue Phasen, neue Musical Goals, API-Änderungen):

```
1. Issue mit Label 'RFC' öffnen
2. Beschreibung: Motivation + Algorithmus + Auswirkung auf Musical Goals
3. Feedback-Periode: mindestens 7 Tage
4. Entscheidung durch Hauptentwickler im Issue dokumentiert
5. Bei Ablehnung: schriftliche Begründung im Issue
```

---

## Beitragsprozess

```
Fork → Feature-Branch → Pre-Commit läuft durch → Tests grün → PR öffnen
```

**Pflicht-Checkliste für jeden PR:**

- [ ] `pre-commit run --all-files` ohne Fehler
- [ ] `pytest tests/unit -q --timeout=30` grün
- [ ] Keine neuen ruff/mypy-Fehler
- [ ] Musical Goals: kein Ziel schlechter als vor dem PR
- [ ] Neues Modul: `≥ 20 Unit-Tests` (Phasen/Plugins) bzw. `≥ 35` (Kernmodule)
- [ ] CHANGELOG.md-Eintrag geschrieben

---

## Plugin-API Versioning

Aurik 10 verpflichtet sich zu **Plugin-API-Stabilität** innerhalb einer Hauptversion (10.x):

```
10.x.y → PhaseInterface, RestorationResult, MusicalGoalsChecker API stabil
11.0   → Breaking Changes erlaubt, Migrationsguide in CHANGELOG.md Pflicht
```

**Deprecation-Policy:**

1. Veraltete API: 2 Minor-Releases lang mit `DeprecationWarning`
2. Dann: Entfernung erst in der nächsten Hauptversion (11.x)
3. Deprecation-Hinweis immer in `CHANGELOG.md` und im Docstring

---

## Code of Conduct

→ [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

---

## Kommunikationskanäle

- **Bugs / Feature-Requests**: GitHub Issues
- **Diskussionen**: GitHub Discussions (falls aktiviert)
- **Sicherheitslücken**: SECURITY.md (private Meldung, kein öffentliches Issue)

---

## Sponsoring & Finanzierung

Aurik 10.0.8 ist kostenlos und Open Source. Freiwillige Spenden sind möglich:
→ [.github/FUNDING.yml](.github/FUNDING.yml)

Gespendete Mittel fließen in:

- Server/Build-Infrastruktur
- ML-Modell-Lizenz-Anfragen (z. B. CC BY-NC-SA → Apache für Gewichte)
- Community-Events

---

_Stand: Juli 2026 — Aurik 10_
