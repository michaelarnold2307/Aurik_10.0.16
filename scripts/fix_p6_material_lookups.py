#!/usr/bin/env python3
"""§v10.113 Auto-fixer: P6 material Dict-Lookup absichern.

Fügt `_mk = material.value if isinstance(material, MaterialType) else material`
vor jedem ungeschützten `[material]` oder `.get(material` ein und ersetzt die
Referenz.

Usage: python3 scripts/fix_p6_material_lookups.py [--dry-run]
"""

import os
import re
import sys

DRY_RUN = '--dry-run' in sys.argv

NORMALIZER = '_mk = material.value if isinstance(material, MaterialType) else material  # §v10.113'


def fix_file(filepath: str) -> int:
    with open(filepath) as f:
        source = f.read()

    if 'MaterialType' not in source:
        return 0

    lines = source.split('\n')
    changes = 0

    for i, line in list(enumerate(lines)):
        s = line.strip()
        if s.startswith('#'):
            continue
        if not (re.search(r'\[material\]', s) or re.search(r'\.get\(material[,\\)]', s)):
            continue

        # Check if already guarded (3 lines before)
        ctx_before = '\n'.join(lines[max(0, i-3):i])
        if 'isinstance(material, MaterialType)' in ctx_before:
            continue
        if 'hasattr(material, "value")' in ctx_before:
            continue
        if '_mat_enum_' in ctx_before:
            continue
        if '.get(_mat_' in ctx_before or '.get(_mk' in ctx_before:
            continue

        # Get indentation of current line
        indent = line[:len(line) - len(line.lstrip())]

        # Find material reference in this line
        if '[material]' in s:
            # Pattern: DICT[material] → DICT.get(_mk, DICT["unknown"])
            # We change this to use .get with fallback
            line_fixed = line.replace('[material]', '.get(_mk, ')
            # Need to find the closing ] and add fallback
            # Actually simpler: replace [material] with [_mk] and add .get fallback
            # For now: just replace [material] with [_mk]
            line_fixed = line.replace('[material]', '[_mk]')
            lines[i] = line_fixed
            # Insert normalizer BEFORE this line
            normalizer_line = f'{indent}{NORMALIZER}'
            lines.insert(i, normalizer_line)
            changes += 1

        elif '.get(material' in s:
            # Pattern: DICT.get(material → DICT.get(_mk
            line_fixed = line.replace('.get(material', '.get(_mk')
            lines[i] = line_fixed
            normalizer_line = f'{indent}{NORMALIZER}'
            lines.insert(i, normalizer_line)
            changes += 1

    if changes > 0:
        if not DRY_RUN:
            # Also add MaterialType import if not present
            new_source = '\n'.join(lines)
            if 'from backend.core.defect_scanner import MaterialType' not in new_source:
                # Add import near existing similar imports
                new_lines = new_source.split('\n')
                insert_pos = 0
                for j, nl in enumerate(new_lines):
                    if 'from backend.core' in nl or 'from backend.core.' in nl:
                        insert_pos = j + 1
                if insert_pos > 0:
                    new_lines.insert(insert_pos, 'from backend.core.defect_scanner import MaterialType  # §v10.113')
                    new_source = '\n'.join(new_lines)

            with open(filepath, 'w') as f:
                f.write(new_source)
            print(f'  ✅ {os.path.basename(filepath)}: {changes} fixes')
        else:
            print(f'  🔍 {os.path.basename(filepath)}: {changes} fixes (dry-run)')
    return changes


def main():
    total = 0
    files_fixed = 0
    for root, dirs, files in os.walk('backend/core'):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if not f.endswith('.py'):
                continue
            fp = os.path.join(root, f)
            c = fix_file(fp)
            if c > 0:
                total += c
                files_fixed += 1

    print(f'\nTotal: {total} fixes in {files_fixed} files')
    if DRY_RUN:
        print('⚠️  DRY RUN — run without --dry-run to apply')


if __name__ == '__main__':
    main()
