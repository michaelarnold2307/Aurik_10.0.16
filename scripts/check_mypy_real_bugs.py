#!/usr/bin/env python3
"""Aurik mypy Real-Bug-Gate.

Fuehrt mypy auf den release-relevanten Ebenen aus und blockiert jeden Fehlercode.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter

DEFAULT_TARGETS = ["backend/core/", "backend/api/", "plugins/", "Aurik10/", "cli/"]
# §10: Keine tolerierten mypy-Fehlercodes in Release-Layern (normativ).
IGNORED_CODES: set[str] = set()
ERROR_CODE_RE = re.compile(r"\[([a-z0-9-]+)\]$")


def run_mypy(targets: list[str]) -> tuple[int, list[str]]:
    command = [
        sys.executable,
        "-m",
        "mypy",
        *targets,
        "--follow-imports=skip",
        "--no-error-summary",
        "--ignore-missing-imports",
        "--show-error-codes",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stdout.splitlines() + result.stderr.splitlines()


def keep_real_bug(line: str) -> bool:
    if ": error:" not in line:
        return False
    match = ERROR_CODE_RE.search(line)
    if match and match.group(1) in IGNORED_CODES:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Aurik mypy Real-Bug-Gate")
    parser.add_argument("targets", nargs="*", default=DEFAULT_TARGETS)
    args = parser.parse_args()

    _, lines = run_mypy(args.targets)
    real_bugs = [line for line in lines if keep_real_bug(line)]
    if not real_bugs:
        print("Aurik mypy Real-Bug-Gate: 0 Fehlercodes in Release-Layern")
        return 0

    counts: Counter[str] = Counter()
    for line in real_bugs:
        match = ERROR_CODE_RE.search(line)
        counts[match.group(1) if match else "unknown"] += 1

    print("Aurik mypy Real-Bug-Gate fehlgeschlagen:")
    for code, count in counts.most_common():
        print(f"  {code}: {count}")
    print("\nErste Befunde:")
    for line in real_bugs[:80]:
        print(line)
    if len(real_bugs) > 80:
        print(f"... {len(real_bugs) - 80} weitere")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
