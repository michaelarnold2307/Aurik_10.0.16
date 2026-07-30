# §FAIR-PAY — Faires Zahlungsmodell für Aurik

> **Status:** Empfehlung · noch nicht implementiert
> **Version:** 2.0 · 2026-08-02
> **Änderung zu v1.0:** Kosten-Transparenz entfernt. Fokus auf qualitätsabhängige Vorschläge.

---

## 1. Ausgangslage

### 1.1 Was Aurik heute ist

| Merkmal | Stand |
|---|---|
| Audio-Restaurierung | Weltspitze — 66 Phasen, Closed-Loop-Optimierung, ML-Hybrid |
| Lizenz | MIT — kostenlos, auch kommerziell |
| Werbung | Keine |
| Monetarisierung | Freiwillige Spende (PayPal) |
| Nutzer-Erlebnis | 1× pro Tag ein freundlicher Spenden-Dialog |

### 1.2 Marktvergleich

| Produkt | Preis | Aurik-Vorteil |
|---|---|---|
| iZotope RX 11 Advanced | 1.199 $ (einmalig) | Aurik: kostenlos, mehr Phasen, adaptiver |
| Acon Digital Restoration Suite | 99 $ | Aurik: KI-basiert, kein manuelles Editieren |
| Waves WNS + X-Noise + X-Crackle | ~350 $ kombiniert | Aurik: alles integriert, automatisch |
| Adobe Audition (Abo) | 22 $/Monat | Aurik: kein Abo, kein Cloud-Zwang |

**Fazit:** Aurik bietet mehr als die kommerziellen Alternativen — und ist kostenlos. Das schafft Vertrauen, aber auch die Frage: „Warum sollte ich zahlen, wenn es umsonst ist?“ Die Antwort: Weil Qualität einen Wert hat.

---

## 2. Das empfohlene Modell: **Aurik Fair-Tier**

### 2.1 Grundprinzip: Vertrauen + Fairness

- **Kein Zwang.** Aurik bleibt funktional unbegrenzt. Niemand wird ausgesperrt.
- **Keine künstliche Verknappung.** Alle 66 Phasen, volle Qualität, für jeden.
- **Die App zeigt, was eine Restaurierung qualitativ wert ist** — und überlässt dem Nutzer die faire Entscheidung.

### 2.2 Die drei Stufen

| Stufe | Name | Empfohlener Beitrag | Zielgruppe |
|---|---|---|---|
| **♡** | Aurik Free | 0 € | Gelegenheitsnutzer, Hobby |
| **★** | Aurik Supporter | 5–10 € / Monat | Regelmäßige Nutzer, Semi-Professionell |
| **✦** | Aurik Pro | 15–25 € / Monat | Profi-Studios, Archivare, kommerzielle Nutzung |

**Wichtig:** Alle Stufen haben **identische Funktionalität**. Der Unterschied ist rein freiwillig — ähnlich wie bei Wikipedia, Blender oder Signal.

### 2.3 Was der Nutzer für seine Unterstützung bekommt

| Feature | Free | Supporter | Pro |
|---|---|---|---|
| Alle 66 Restaurierungsphasen | ✓ | ✓ | ✓ |
| Volle 48-kHz-Qualität | ✓ | ✓ | ✓ |
| Batch-Verarbeitung | ✓ | ✓ | ✓ |
| Spenden-Dialog (1×/Tag) | ✓ | — | — |
| Name im About-Dialog | — | ✓ | ✓ |
| Priorisierter Feature-Wunsch | — | ✓ | ✓ |
| Early Access auf Beta-Versionen | — | — | ✓ |
| Direkter E-Mail-Support | — | — | ✓ |
| „Aurik Pro“-Zertifikat (PDF) | — | — | ✓ |

---

## 3. Qualitätsabhängige Vorschläge

### 3.1 Kernidee

Statt „bitte spende“ zeigt Aurik:

> **„Dein Song wurde mit Qualität 87/100 restauriert. Ein fairer Beitrag wäre 3 €.“**

Das System kennt den objektiven Quality Score jeder Restaurierung und schlägt daraus einen nachvollziehbaren Betrag vor — ohne Druck, ohne Zwang.

### 3.2 Formel für den fairen Vorschlag

```
vorschlag = base × quality_factor × loyalty_factor
```

| Parameter | Berechnung |
|---|---|
| `base` | 5,00 € (Basisbetrag pro Restaurierung) |
| `quality_factor` | `clamp(quality_score / 100, 0,25, 1,0)` |
| `loyalty_factor` | 1,0 bei 1–4/Monat; 0,75 bei 5–19/Monat; 0,60 bei ≥20/Monat |

### 3.3 Qualitätsstufen und ihre Vorschläge

| Quality Score | Bewertung | Vorschlag (1. Song/Monat) | Vorschlag (10. Song/Monat) |
|---|---|---|---|
| ≥ 85 | 🌟 Hervorragend — Weltspitze-Ergebnis | 4,25 € | 3,19 € |
| 70–84 | ✨ Sehr gut — deutliche Verbesserung | 3,50 € | 2,63 € |
| 50–69 | 👍 Gut — spürbar restauriert | 2,50 € | 1,88 € |
| 30–49 | 🎧 Solide — Aurik hat sein Bestes gegeben | 1,50 € | 1,13 € |
| < 30 | 🎵 Basis-Restaurierung | 1,25 € | 0,94 € |

### 3.4 Pay-what-you-want: Einmalzahlungen

Nicht jeder will monatlich spenden:

| Paket | Preis | Enthalten |
|---|---|---|
| Einzeltrack | 2–5 € (frei wählbar) | 1 Restaurierung |
| 10er-Pack | 15 € | 10 Restaurierungen (1,50 €/Stück) |
| 50er-Pack | 49 € | 50 Restaurierungen (0,98 €/Stück) |
| Jahres-Flat | 69 € | Unbegrenzt für 12 Monate |
| Lifetime | 199 € | Unbegrenzt, lebenslang |

### 3.5 Niedrige Einstiegsschwelle

- Die ersten **3 Songs sind komplett frei** — kein Dialog, kein Hinweis.
- Der Spenden-Dialog erscheint erst beim 4. Song und dann **maximal 1× pro Tag**.
- Der Dialog hat immer einen gut sichtbaren **„Später“-Button** (schließt für 48 Stunden).
- Bei Quality Score < 30 erscheint der Dialog seltener (nur alle 5 abgeschlossene Songs).

---

## 4. Implementierungs-Roadmap (Reihenfolge)

### Phase 1 — Jetzt ✔️
- Freundlicher Spenden-Dialog nach Batch-Abschluss
- PayPal-Button zu michael.arnold2307@gmail.com
- Rate-Limit: 1× pro 24 Stunden

### Phase 2 — Qualitäts-Vorschlag (Aufwand: ~2 Tage)
- `donation_reminder.py` → neues Modul `backend/core/fair_payment.py`
- Funktion `compute_fair_price(quality_score, usage_count) -> FairPriceSuggestion`
- GUI: Dialog zeigt Quality-Score + personalisierten Vorschlag
- Drei Buttons: Vorgeschlagener Betrag, doppelter Betrag, „Später“

### Phase 3 — Supporter-Features (Aufwand: ~5 Tage)
- Lizenzschlüssel-System (offline-validiert, HMAC-signiert)
- `backend/core/licensing.py`: `LicenseTier(FREE, SUPPORTER, PRO)`
- Supporter: kein Spenden-Dialog, Name im About, priorisierte Feature-Requests
- Pro: zusätzlich Beta-Channel, Zertifikat, Direktsupport

### Phase 4 — Bezahl-Plattform (Aufwand: ~10 Tage)
- Integration mit Stripe / LemonSqueezy / Gumroad
- Monatliche / jährliche / Lifetime-Pläne
- Automatische Lizenzschlüssel-Generierung nach Zahlung
- Webhook für Zahlungsbestätigung

---

## 5. Kommunikation mit dem Nutzer

### Der qualitätsabhängige Dialog (Phase 2, Entwurf):

```
🌟 Hervorragende Restaurierung — Qualität 87/100

Aurik hat Deinen Song auf Weltspitze-Niveau gebracht.
Ein fairer Beitrag für diese Qualität: 4,25 €

[💛 4 € spenden]  [💙 8 € spenden]  [🤍 Später]

Du hast diesen Monat schon 8 Songs restauriert.
Als regelmäßiger Nutzer sparst Du 25 % gegenüber Einzelspenden.

— Michael (Aurik-Entwickler)
```

Bei Quality Score < 30:
```
🎵 Basis-Restaurierung — Qualität 27/100

Das Ausgangsmaterial war schwierig. Aurik hat sein Bestes gegeben.
Falls Dir das Ergebnis trotzdem hilft, freue ich mich über jede Spende.

[💛 1 € spenden]  [💙 3 € spenden]  [🤍 Später]

— Michael (Aurik-Entwickler)
```

---

## 6. Risiken & Ethische Leitplanken

| Risiko | Maßnahme |
|---|---|
| Nutzer fühlen sich unter Druck gesetzt | Dialog hat immer „Später“-Button. Erste 3 Songs dialogfrei. |
| Open-Source-Community rebelliert | MIT-Lizenz garantiert Fork-Recht. Aurik bleibt Open Source. |
| Zahlungsausfälle bei Abos | Keine automatische Sperre — nur Hinweis beim Start. |
| Missbrauch (Cracking) | Lizenzprüfung ist HMAC-signiert, aber nicht obfuskiert. |
| Datenschutz (DSGVO) | Keine zwingende Registrierung. PayPal-Daten verlassen nie den Browser. |

---

## 7. Fazit & Nächster Schritt

Das Fair-Tier-Modell respektiert den Open-Source-Geist von Aurik und macht die Qualität der Restaurierung zur Grundlage eines fairen, freiwilligen Beitrags.

**Die wichtigste Regel: Aurik nie hinter eine Paywall setzen.**

➡️ **Nächster konkreter Schritt: Phase 2 — `compute_fair_price()` implementieren.**
