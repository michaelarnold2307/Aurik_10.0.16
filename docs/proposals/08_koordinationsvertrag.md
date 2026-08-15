# Vorschlag 08 — Koordinationsvertrag für parallele Agenten

> **Status: VORSCHLAG (Entwurf) — nicht normativ.** Übernahme als
> `AGENTS.md` §9 (Verhaltensregeln, als solche gekennzeichnet) +
> Checkliste im PR-Template.
> Formel und Meta-Regel: `00_interpretationsfrei_uebersicht.md`.

## Zweck

Mess-Evidenz aus der Session: Zwei Agenten im selben Arbeitsbaum führten
zu (a) einem **verlorenen uncommitteten WIP-File** (`git checkout --` im
falschen Baum bei laufender Parallel-Sitzung), (b) vermischten
Testzuständen während laufender Vollsuite-Läufe, (c) einem abgebrochenen
Commit durch konkurrierende Git-Operationen. Verhaltensregeln ohne
Maschinenprüfung sind interpretationsanfällig — deshalb wird hier
explizit markiert, was maschinell prüfbar ist und was menschlich bleibt.

## Normativer Wortlaut (für AGENTS.md §9)

> **§9 Parallele Agenten (Verhaltensvertrag — maschinell nur teilweise
> prüfbar):**
>
> 1. **Datei-Cluster-Eigentum**: Jeder Agent deklariert zu Beginn einer
>    Aufgabe seinen Datei-Cluster (Log/Status-Nachricht). Fremde,
>    uncommittete Dateien werden NIE editiert oder committet.
> 2. **Baseline-Experimente nur in eigenen Worktrees**
>    (`git worktree add /tmp/<agent>-<id>`), niemals im Hauptbaum.
> 3. **HARTER VERBOT**: `git checkout --`, `git reset --hard`,
>    `git stash drop` im Hauptbaum, solange eine parallele Sitzung
>    aktiv ist. Zuwiderhandlung ist sofort und vollständig zu melden.
> 4. **Commit-Atomizität**: Kleine, thematische Commits; `git add` nur
>    der eigenen Dateien; vor jedem Commit `git status` auf fremde
>    Änderungen prüfen.
> 5. **Prüfbarer Teil**: Ein Pre-Commit-Hook
>    (`aurik-parallel-safety`, aktivierbar via `AURIK_PARALLEL=1`) lehnt
>    Commits ab, deren gestagte Dateien jünger als die letzte fremde
>    Sitzungs-Markierung `docs/.agent-owner.json` sind und nicht dem
>    eigenen Cluster angehören.

## Enforcement

1. **Maschinell (optional aktiv)**: `docs/.agent-owner.json`
   (Agent-ID → Datei-Cluster, mtime der letzten fremden Aktivität) +
   Hook `aurik-parallel-safety`.
2. **Menschlich (Checkliste im PR-Template)**:
   `- [ ] Keine fremden uncommitteten Dateien übernommen`
   `- [ ] Keine `git checkout --`/`reset --hard` im Hauptbaum während
     paralleler Sitzungen`
3. Session-Lektion als Test verankert: `tests/normative/
   test_worktree_isolation_convention.py` prüft, dass der §9-Text die
   drei Verbote wörtlich enthält (Konsistenz-Wächter, kein
   Verhaltensbeweis).

## Entfernter Spielraum

- Die Frage „darf ich in fremden Dateien arbeiten?" (klare Regeln +
  Checkliste + optionaler Hook).
- Wann Experimente in Worktrees gehören (immer).
- Die destruktiven Git-Befehle im Hauptbaum (hartes Verbot).

## Betroffene Dateien

- `AGENTS.md` (§9)
- `.github/pull_request_template.md` (Checkliste)
- `.pre-commit-config.yaml` + `scripts/aurik_parallel_safety.py`
  (optionaler Hook, NEU)
- `tests/normative/test_worktree_isolation_convention.py` (NEU)

## Risiken

- Verhaltensregeln sind ohne Willen nicht durchsetzbar — deshalb die
  ehrliche Kennzeichnung: dieser Vorschlag eliminiert Spielraum durch
  Eindeutigkeit und Nachprüfbarkeit, nicht durch Zwang.
