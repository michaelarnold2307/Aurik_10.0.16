# Vorschlag 04 — STFT-Konvention: Ein Paar, kein Global-Patch

> **Status: VORSCHLAG (Entwurf) — nicht normativ.** Übernahme als
> `VERBOTEN.md` V54 + Umsetzung in `backend/__init__.py` / `audio_utils.py`.
> Formel und Meta-Regel: `00_interpretationsfrei_uebersicht.md`.

## Zweck

Mess-Evidenz aus der Session: Der globale `scipy.signal.stft`-Monkeypatch
in `backend/__init__.py` erzeugte in **vier** Modulen frame-inkonsistente
STFT/ISTFT-Paare (Console-EQ, SGMSE+, hybrid_ml_denoiser,
spectral_denoiser) mit entkoppelten bzw. verschobenen Ausgaben. Die Spec
kündigt die Entfernung des Patches für „v10.18" an — ohne Datum und ohne
Lint-Schutz ist das interpretationsanfällig.

## Normativer Wortlaut (für VERBOTEN.md V54)

> `[RELEASE_MUST] §V54 STFT-KONVENTION: Direkte Aufrufe von
> `scipy.signal.stft` / `scipy.signal.istft` in `backend/`, `plugins/`
> und `dsp/` sind VERBOTEN. Einzige erlaubte Paarung ist
> `backend.core.audio_utils.safe_stft` / `safe_istft`, wobei
> `boundary` in BEIDEN Aufrufen explizit identisch gesetzt sein MUSS
> (kanonisch `boundary="zeros"`).
>
> Der globale Monkeypatch `_scipy_signal.stft = _safe_stft` in
> `backend/__init__.py` wird im Release **v10.18 (2026-09-30)**
> ersatzlos entfernt. Bis dahin MUSS der Patch transparent sein
> (identische Frame-Zahlen wie vanilla scipy bei gleichen Argumenten —
> nachweisbar über den unten genannten Test). Nach der Entfernung ist
> jede Wiedereinführung eines globalen SciPy-Patches VERBOTEN.

## Enforcement

1. **Lint (fail-closed)** in `aurik-verboten-linter`:
   `SCIPY_STFT_DIRECT` — `scipy.signal.stft(`/`istft(` außerhalb von
   `audio_utils.py` und `backend/__init__.py` ⇒ ERROR.
2. `[RELEASE_MUST]`-Tests `tests/unit/test_stft_konvention.py`:
   - `test_patch_frame_parity`: gepatchter stft(boundary="zeros") liefert
     identische Frame-Zahl wie vanilla scipy (vor v10.18).
   - `test_roundtrip_identity_after_backend_import`: sinus/noise-Roundtrip
     MAE < 1e-6 nach `import backend`.
   - `test_no_global_patch_after_v10_18`: ab Release-Zweig v10.18 schlägt
     der Test fehl, wenn `_scipy_signal.stft` noch gepatcht ist.
3. **Kalender-Gate**: `scripts/check_deprecation_deadlines.py` liest das
   Datum aus V54 und warnt ab 2026-09-01, blockiert ab 2026-10-01.

## Entfernter Spielraum

- Patch vs. kein Patch (nur noch eine Übergangsregel mit Frist).
- boundary-Wahl (kanonisch "zeros", Lint prüft Konsistenz).
- „Alle Phasen verwenden explizit safe_stft" ⇒ maschinell erzwungen.

## Betroffene Dateien

- `backend/__init__.py` (Patch bleibt bis v10.18, dann entfernt)
- `scripts/aurik_verboten_linter.py` (Regel)
- `scripts/check_deprecation_deadlines.py` (NEU)
- Migration verbliebener Direktaufrufer (per Lint-Report ermittelt)
- `tests/unit/test_stft_konvention.py` (NEU)

## Risiken

- Die Frist ist eine Wette auf Kalenderstabilität — Gegenmaßnahme:
  Warn-Phase ab 30 Tage vorher; die Entfernung ist ein eigener Commit
  mit Evidenzblock.
