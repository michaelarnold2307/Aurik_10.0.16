#!/usr/bin/env python3
"""§v10.113 P6 Fixer v2 — konservativ, dateiweise, compile-verifiziert.

Nur zwei Operationen:
1. Vor dem ERSTEN `.get(material` pro Funktionsblock: `_mk = ...` einfügen
2. ALLE `.get(material` → `.get(_mk` im selben Block

NIEMALS: Import-Insert, Bracket-Änderung, Multi-Line-Umbau.
"""

import os
import re
import sys
import py_compile

NORMALIZER = '_mk = material.value if isinstance(material, MaterialType) else material  # §v10.113'


def find_functions_with_material(source: str):
    """Finde alle Funktionen, die material-Dict-Lookups enthalten."""
    lines = source.split('\n')
    # Finde Funktions-Starts (def ...) und ihre Einrückung
    func_starts = []
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r'^\s*def\s+\w+\s*\(', line):
            indent = len(line) - len(line.lstrip())
            func_starts.append((i, indent))

    if not func_starts:
        # Top-level code: treat whole file as one block
        return [(0, [])]

    # Finde Lookups und ordne sie Funktionen zu
    results = []
    for fi, (start, base_indent) in enumerate(func_starts):
        end = func_starts[fi+1][0] if fi+1 < len(func_starts) else len(lines)
        lookups = []
        for i in range(start, end):
            s = lines[i].strip()
            if s.startswith('#'):
                continue
            if re.search(r'\.get\(material[,\\)]', s):
                ctx = '\n'.join(lines[max(0,i-3):i])
                if not any(x in ctx for x in ['_mk', '_mat_enum_', 'isinstance(material, MaterialType)']):
                    lookups.append(i)
        if lookups:
            results.append((start, base_indent, lookups))

    return results


def fix_file(filepath: str) -> bool:
    with open(filepath) as f:
        source = f.read()

    if 'MaterialType' not in source:
        return False

    blocks = find_functions_with_material(source)
    if not blocks:
        return False

    lines = source.split('\n')
    offset = 0  # track line shifts from insertions

    for _, base_indent, lookup_lines in blocks:
        if not lookup_lines:
            continue

        # Determine indentation: find actual indent at first lookup
        first = lookup_lines[0] + offset
        actual_line = lines[first]
        indent = actual_line[:len(actual_line) - len(actual_line.lstrip())]

        # Insert normalizer BEFORE first lookup
        normalizer_line = f'{indent}{NORMALIZER}'
        lines.insert(first, normalizer_line)
        offset += 1

        # Now replace ALL .get(material, with .get(_mk, in this block
        for li in lookup_lines:
            adjusted = li + offset
            if adjusted < len(lines):
                lines[adjusted] = lines[adjusted].replace('.get(material,', '.get(_mk,')

    new_source = '\n'.join(lines)
    if new_source == source:
        return False

    with open(filepath, 'w') as f:
        f.write(new_source)
    return True


def main():
    fixed = 0
    failed = []

    targets = []
    for root, dirs, files in os.walk('backend/core'):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if not f.endswith('.py'):
                continue
            targets.append(os.path.join(root, f))

    for fp in sorted(targets):
        try:
            changed = fix_file(fp)
            if changed:
                # Verify compile
                try:
                    py_compile.compile(fp, doraise=True)
                    print(f'  ✅ {os.path.basename(fp)}')
                    fixed += 1
                except py_compile.PyCompileError as e:
                    print(f'  ❌ {os.path.basename(fp)}: COMPILE ERROR — rolling back')
                    # Git restore
                    os.system(f'git checkout -- {fp}')
                    failed.append(fp)
        except Exception as e:
            print(f'  ⚠️ {os.path.basename(fp)}: {e}')

    print(f'\nFixed: {fixed}, Failed (rolled back): {len(failed)}')
    if failed:
        print('Failed files:')
        for f in failed:
            print(f'  {f}')


if __name__ == '__main__':
    main()
