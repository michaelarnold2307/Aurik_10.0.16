# Vorschlag 06 — Statische Gates wirksam machen (fail-closed auf gestagten Dateien)

> **Status: VORSCHLAG (Entwurf) — nicht normativ.** Übernahme in
> `.github/copilot-instructions.md` (§Performance/CI) +
> `.pre-commit-config.yaml`.
> Formel und Meta-Regel: `00_interpretationsfrei_uebersicht.md`.

## Zweck

Mess-Evidenz aus der Session: (a) **5 doppelte `except`-Klauseln** (B025)
in einer Session (VQI, phase_12 ×2, UV3) — die zweite Klausel ist toter
Code und maskiert Fallback-Bugs; (b) ein **fehlendes `return`** in
`validate_final_quality`, maskiert durch `# type: ignore[return]`;
(c) die Pre-Commit-Ausgabe zeigte bei mehreren Commits
„ruff … (no files to check) Skipped" — die statischen Gates liefen
nachweislich nicht auf den gestagten Dateien.

## Normativer Wortlaut (für copilot-instructions)

> `[RELEASE_MUST] §CI-STATIC-GATES:
>
> 1. Alle statischen Pre-Commit-Gates MÜSSEN auf den Dateien von
>    `git diff --cached` laufen; ein Gate, das `(no files to check)`
>    meldet, obwohl gestagte Python-Dateien existieren, ist als
>    Hook-Fehler zu behandeln (Commit wird abgelehnt).
> 2. **B025** (try-except mit doppelter Exception-Klasse) ist Teil des
>    fail-closed-Katalogs (`ruff select` enthält B) und darf in keinem
>    Commit erscheinen.
> 3. `# type: ignore` ist nur mit Begründungskommentar im selben
>    logischen Ausdruck zulässig; `# type: ignore[return]` auf Funktionen
>    mit Rückgabewert ist VERBOTEN (die Funktion MUSS returnen oder als
>    `-> None` deklariert sein).
> 4. Der Hook-Selbsttest `scripts/test_precommit_hooks.sh` MUSS bei
>    Release-Evidenz laufen: ein absichtlicher B025-Commit wird abgelehnt
>    (Fail-Fall) und ein sauberer Commit akzeptiert (Pass-Fall).

## Enforcement

1. `.pre-commit-config.yaml`: `files:`-Filter der `aurik-*`-Gates auf
   gestagte Dateien + Fail-When-Skipped-Mechanik.
2. `scripts/test_precommit_hooks.sh` (NEU): erzeugt einen Throwaway-Commit
   mit B025 in einem Temp-Repo-Clone und prüft Ablehnung.
3. Lint: B025 in den fail-closed-Katalog aufnehmen (heute nur „legacy
   alias"-Präsenz).

## Entfernter Spielraum

- Ob statische Gates liefen (maschinell nachweisbar statt Log-Lesart).
- Doppelte except-Klauseln (verboten statt „kann man zusammenführen").
- Maskierte Rückgabefehler (Ignore-Verbot + Rückgabetyp-Konsistenz).

## Betroffene Dateien

- `.pre-commit-config.yaml`
- `.github/copilot-instructions.md` (Regeltext)
- `scripts/test_precommit_hooks.sh` (NEU)
- `scripts/aurik_verboten_linter.py` / ruff-Config (B025-Katalog)

## Risiken

- Bestandscode enthält weitere B025-Fälle — Gegenmaßnahme: Einmal-
  Bereinigung vor Aktivierung, danach fail-closed (die Session hat bereits
  5 Fälle gefixt; Rest per `ruff check .` ermitteln).
