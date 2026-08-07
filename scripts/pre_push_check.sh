#!/usr/bin/env bash
# scripts/pre_push_check.sh — §v10.700
#
# Schnelle Pre-Push-Validierung (<60s).
# Läuft vor jedem git push. Blockt bei Fehlern.
#
# Nutzung:
#   bash scripts/pre_push_check.sh          # Kurz-Check
#   bash scripts/pre_push_check.sh --full   # Vollständiger Check
#   bash scripts/pre_push_check.sh --ci     # CI-Mode (JSON-Output)
#
# Exit 0 = OK, Exit 1 = Fehler.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RESET='\033[0m'

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=".venv_aurik/bin/python"
FULL_CHECK=false
CI_MODE=false
ERRORS=0

for arg in "$@"; do
    case "$arg" in
        --full) FULL_CHECK=true ;;
        --ci) CI_MODE=true ;;
    esac
done

ok() { echo -e "${GREEN}✅ $1${RESET}"; }
fail() { echo -e "${RED}❌ $1${RESET}"; ERRORS=$((ERRORS + 1)); }

# ── 1. Quick Smoke Test ──────────────────────────────────────────
echo -e "\n${YELLOW}═══ 1. Quick Smoke Test ═══${RESET}"
if $PYTHON scripts/ci_quick_smoke.py; then
    ok "Quick Smoke"
else
    fail "Quick Smoke"
fi

# ── 2. Version Consistency ───────────────────────────────────────
echo -e "\n${YELLOW}═══ 2. Version Consistency ═══${RESET}"
if $PYTHON scripts/check_version_consistency.py; then
    ok "Version Consistency"
else
    fail "Version Consistency"
fi

# ── 3. Compliance ────────────────────────────────────────────────
echo -e "\n${YELLOW}═══ 3. Compliance (VERBOTEN) ═══${RESET}"
if $PYTHON scripts/compliance_check.py --errors-only; then
    ok "Compliance"
else
    fail "Compliance"
fi

# ── 4. Doc Consistency (F7) ──────────────────────────────────────
echo -e "\n${YELLOW}═══ 4. Doc Consistency ═══${RESET}"
if $PYTHON scripts/check_doc_consistency.py --ci; then
    ok "Doc Consistency"
else
    fail "Doc Consistency"
fi

# ── Optional: Full mode ──────────────────────────────────────────
if $FULL_CHECK || $CI_MODE; then
    echo -e "\n${YELLOW}═══ 5. Unit Tests (full mode) ═══${RESET}"
    if bash scripts/pytest_clean.sh tests/unit \
        -p no:xdist \
        --override-ini="addopts=--strict-markers --import-mode=importlib" \
        --timeout=30 --tb=short -q --disable-warnings --no-header; then
        ok "Unit Tests"
    else
        fail "Unit Tests"
    fi

    echo -e "\n${YELLOW}═══ 6. Normative Guard ═══${RESET}"
    if $PYTHON -m pytest tests/normative/test_no_production_stubs.py \
        -p no:xdist \
        --override-ini="addopts=--strict-markers --import-mode=importlib" \
        --timeout=30 --tb=short -q --disable-warnings --no-header; then
        ok "Normative Guard"
    else
        fail "Normative Guard"
    fi
fi

# ── Ergebnis ─────────────────────────────────────────────────────
echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════${RESET}"
    echo -e "${GREEN} ✅ ALLE CHECKS BESTANDEN — Push bereit${RESET}"
    echo -e "${GREEN}═══════════════════════════════════${RESET}"
    exit 0
else
    echo -e "${RED}═══════════════════════════════════${RESET}"
    echo -e "${RED} ❌ $ERRORS CHECK(S) FEHLGESCHLAGEN — Push blockiert${RESET}"
    echo -e "${RED}═══════════════════════════════════${RESET}"
    exit 1
fi
