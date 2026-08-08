#!/usr/bin/env python3
"""
Fix: silent except blocks in unified_restorer_v3.py (§2.59 Muster 2)

Adds logger.debug(..., exc_info=True) before bare pass/return/continue
in except Exception: blocks that have no logging.
"""

import re
import sys
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
TARGET = ROOT / "backend" / "core" / "unified_restorer_v3.py"

if not TARGET.exists():
    print(f"File not found: {TARGET}")
    sys.exit(1)

source = TARGET.read_text(encoding="utf-8")
lines = source.split("\n")
new_lines = list(lines)

# Pattern to find the function/context name
FUNC_PATTERN = re.compile(r"^\s{4}def (\w+)")

changes = 0
i = 0
while i < len(lines):
    line = lines[i]
    # Match "except Exception:" or "except Exception as ...:"
    if re.match(r"\s*except\s+Exception\s*(as\s+\w+)?\s*:", line):
        indent = len(line) - len(line.lstrip())
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        # Check if next line is bare pass/return/continue
        if re.match(r"\s*(pass|return|continue)\s*$", next_line):
            # Check if there's a logger call within 3 lines above
            has_logger = False
            for j in range(max(0, i - 3), i):
                if "logger." in lines[j]:
                    has_logger = True
                    break

            if not has_logger:
                # Find the enclosing function name
                func_name = "unknown"
                for j in range(i, -1, -1):
                    m = FUNC_PATTERN.match(lines[j])
                    if m:
                        func_name = m.group(1)
                        break

                # Also try to get a descriptive context from comments
                context = ""
                for j in range(i - 1, max(0, i - 5), -1):
                    if "#" in lines[j] and len(lines[j].strip()) > 3:
                        comment = lines[j].strip().lstrip("#").strip()
                        if len(comment) > 10 and not comment.startswith("noqa"):
                            context = comment[:80]
                            break

                log_indent = " " * (indent + 4)
                if context:
                    log_msg = f'{log_indent}logger.debug("{func_name}: silent except — {context}", exc_info=True)'
                else:
                    log_msg = f'{log_indent}logger.debug("{func_name}: silent except suppressed", exc_info=True)'

                # Insert the log line before the pass/return/continue
                new_lines.insert(i + 1 + changes, log_msg)
                changes += 1
    i += 1

if changes == 0:
    print("No silent except blocks found.")
    sys.exit(0)

# Write back
TARGET.write_text("\n".join(new_lines), encoding="utf-8")
print(f"Fixed {changes} silent except blocks in {TARGET.name}")
print("Added logger.debug(...) with exc_info=True before bare pass/return/continue.")
