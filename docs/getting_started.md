# 🚀 Aurik — Getting Started (10 Minuten)

**Willkommen bei Aurik — dem weltweit führenden Musik-Restaurierungssystem.**

Diese Anleitung bringt Dich in maximal 10 Minuten zu Deiner ersten
erfolgreichen Restaurierung. Keine Vorkenntnisse nötig.

---

## Schritt 1: Aurik herunterladen (1 Minute)

Wähle Dein Betriebssystem:

| Betriebssystem | Download | Anleitung |
|---|---|---|
| **Windows** | `Aurik_Setup_10.14.0.exe` | Doppelklick → „Weiter" → „Fertig" |
| **macOS** | `Aurik_10.14.0.dmg` | DMG öffnen → Aurik in „Applications" ziehen |
| **Linux** | `Aurik-10.14.0-x86_64.AppImage` | `chmod +x` → Doppelklick |

> 📥 **Download**: https://github.com/aurik-audio/Aurik_Standalone/releases/latest

---

## Schritt 2: Aurik starten (1 Minute)

- **Windows**: Doppelklick auf das Aurik-Icon auf dem Desktop
- **macOS**: Aurik im Launchpad oder Programme-Ordner öffnen
- **Linux**: Doppelklick auf die `.AppImage`-Datei

> ⚠️ **macOS-Nutzer**: Beim ersten Start: „Trotzdem öffnen" (Systemeinstellungen → Sicherheit)

Du siehst jetzt den Aurik-Startbildschirm mit dem Titel **„Aurik Professional"**.

---

## Schritt 3: Audio-Datei öffnen (2 Minuten)

1. Klicke auf **„📂 Datei öffnen"** (oben links)
2. Wähle eine Audio-Datei aus (`.wav`, `.mp3`, `.flac`, `.aac`, `.ogg`)
3. Aurik analysiert die Datei automatisch — das dauert wenige Sekunden

Du siehst jetzt:
- Welches **Material** erkannt wurde (Vinyl, Kassette, Shellac…)
- Welche **Defekte** gefunden wurden (Knackser, Rauschen, Brummen…)
- Eine **Qualitäts-Prognose** (z.B. „Gut restaurierbar: 64%")

---

## Schritt 4: Restaurierung starten (1 Minute)

1. Wähle den **Modus**:
   - **Restoration**: Originalgetreue Restaurierung (empfohlen für Vinyl/Tape/Shellac)
   - **Studio 2026**: Moderner High-End-Klang (für gut erhaltenes Material)
2. Klicke auf **„✨ Magic Button"** (der große Button in der Mitte)

Die Restaurierung läuft jetzt — Du siehst einen Fortschrittsbalken und
die aktuelle Phase (z.B. „Rausch-Unterdrückung", „Knackser-Entfernung").

> ⏱ **Dauer**: 5–60 Minuten, je nach Länge und Zustand der Aufnahme.

---

## Schritt 5: Ergebnis anhören & exportieren (2 Minuten)

Nach Abschluss:

1. **Anhören**: Die Buttons „▶ Original" und „▶ Restauriert" spielen
   beide Versionen ab, damit Du den Unterschied hörst.
2. **Exportieren**: Wähle das Format (FLAC empfohlen) und klicke auf
   **„💾 Exportieren"**.

> 🎧 **Tipp**: Drücke die **Leertaste** für Play/Pause.
> Klicke auf die Wellenform, um an eine bestimmte Stelle zu springen.

---

## Fertig! 🎉

Deine restaurierte Datei liegt jetzt im Export-Ordner.

**Was als Nächstes?**
- [Tutorial: Vinyl restaurieren](tutorials/tutorial_restore_vinyl.md)
- [Tutorial: Kassette restaurieren](tutorials/tutorial_restore_tape.md)
- [Tutorial: Batch-Verarbeitung](tutorials/tutorial_batch_processing.md)
- [Alle Funktionen im Überblick](guides/USER_GUIDE.md)

---

## 🐛 Probleme?

| Problem | Lösung |
|---|---|
| „Aurik startet nicht" | [Installationshilfe](guides/INSTALLATION.md) |
| „Keine GPU erkannt" | Das ist normal — Aurik läuft auch auf der CPU |
| „Export schlägt fehl" | [Troubleshooting](guides/TROUBLESHOOTING.md) |
| Datei klingt schlechter als vorher | Schalte auf **„Restoration"-Modus** (nicht Studio 2026) |

---

## 🐳 Docker (für Fortgeschrittene)

```bash
docker run -v "$PWD/audio:/audio" aurik/aurik:10.14.0 restore /audio/mein_song.wav
```

Keine Installation nötig — alle Abhängigkeiten sind im Container.

---

**Aurik 10.14.0 „Durchblick"** — Weltklasse-Audio-Restaurierung für das menschliche Ohr.
