#!/bin/bash
# scripts/sign_release.sh — §v10.700 I3: GPG-Signatur für Linux AppImage.
# Nutzung: ./scripts/sign_release.sh Aurik-10.14.0-x86_64.AppImage
# Voraussetzung: GPG-Schlüssel in ~/.gnupg/

set -e

if [ $# -lt 1 ]; then
    echo "Nutzung: $0 <AppImage-Datei>"
    exit 1
fi

APPIMAGE="$1"
SIG="${APPIMAGE}.sig"

if ! command -v gpg &> /dev/null; then
    echo "❌ GPG nicht installiert — überspringe Signierung"
    exit 0
fi

echo "🔏 Signiere $APPIMAGE ..."
gpg --detach-sign --armor "$APPIMAGE"
echo "✅ Signatur: ${SIG}"
echo "   Prüfen mit: gpg --verify ${SIG} ${APPIMAGE}"
