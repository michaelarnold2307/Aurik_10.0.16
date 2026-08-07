# Aurik Signatur-Prüfung — §v10.700 I3

So prüfst Du, ob Dein Aurik-Download authentisch ist.

## Linux (AppImage)

```bash
# Signatur prüfen
gpg --verify Aurik-10.14.0-x86_64.AppImage.sig Aurik-10.14.0-x86_64.AppImage

# Ausgabe bei gültiger Signatur:
# gpg: Signature made ...
# gpg: Good signature from "Aurik Team"
```

## Windows (Installer)

```powershell
# Im Verzeichnis der heruntergeladenen Datei:
Get-AuthenticodeSignature Aurik_Setup_10.14.0.exe | Format-List

# Ausgabe bei gültiger Signatur:
# Status: Valid
# SignerCertificate: CN=Aurik Audio
```

## macOS (DMG)

Apple Notarization wird automatisch von Gatekeeper geprüft.
Kein manueller Schritt nötig — macOS blockt nicht-notarisierte Apps.

Bei Problemen: Systemeinstellungen → Sicherheit → "Trotzdem öffnen"
