#!/usr/bin/env python3
"""
§IMPORT-GUARD: Prevents import shadowing (UnboundLocalError) in Aurik codebase.

Bug pattern detected 2026-07-22:
  import os                               # module level
  def foo():
      os.path.join(...)                   # UnboundLocalError!
      ...
      import tempfile, os                 # shadows module-level 'os'

This script scans all non-test, non-venv Python files and reports
functions that import a name which is already available at module level,
where the name is used in the function body BEFORE the local import.

Usage:
  python3 scripts/lint_import_shadow.py          # report all
  python3 scripts/lint_import_shadow.py --strict # exit 1 on any finding
"""

import os, re, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {
    '__pycache__', 'models', 'logs', 'output_audio', 'sessions',
    'data', '.venv_aurik', 'venv_rocm', '.venv', 'node_modules', '.git',
    'tests', 'scripts', 'benchmarks',
}

def find_import_shadowing(filepath):
    """Find functions that shadow a module-level import."""
    bugs = []
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except Exception:
        return bugs
    
    # Collect module-level imported names
    module_names = {}
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        indent = len(line) - len(line.lstrip())
        if indent > 0:
            break  # past module-level imports
        m = re.match(r'^import\s+(\w[\w.]*)(?:\s+as\s+(\w+))?', stripped)
        if m:
            name = m.group(2) if m.group(2) else m.group(1).split('.')[0]
            module_names[name] = i
            continue
        m = re.match(r'^from\s+[\w.]+\s+import\s+(.+)', stripped)
        if m:
            for part in m.group(1).split(','):
                part = part.strip()
                if part == '*':
                    continue
                alias_match = re.match(r'(\w+)\s+as\s+(\w+)', part)
                if alias_match:
                    module_names[alias_match.group(2)] = i
                else:
                    module_names[part] = i
    
    if not module_names:
        return bugs
    
    # Find functions with local imports that shadow module-level names
    in_func = False
    func_name = ""
    func_start = 0
    func_indent = 0
    
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        indent = len(line) - len(line.lstrip())
        
        if not in_func:
            if stripped.lstrip().startswith('def ') and indent < 8:
                in_func = True
                func_name = stripped.split('(')[0].replace('def ', '').strip()
                func_start = i
                func_indent = indent
            continue
        
        if stripped and indent <= func_indent:
            in_func = False
            continue
        
        # Check for local import that shadows module-level name
        m = re.match(r'\s*import\s+(\w[\w.]*)(?:\s+as\s+(\w+))?', stripped)
        shadow_name = None
        if m:
            base_name = m.group(1).split('.')[0]
            alias = m.group(2)
            if alias and alias in module_names:
                shadow_name = alias
            elif not alias and base_name in module_names:
                shadow_name = base_name
        
        # Also check `import a, b, c` pattern
        if not shadow_name:
            m = re.match(r'\s*import\s+(.+)', stripped)
            if m:
                imports = m.group(1).split(',')
                for imp in imports:
                    imp = imp.strip()
                    parts = imp.split()
                    if len(parts) == 3 and parts[1] == 'as':
                        name = parts[2]
                    else:
                        name = parts[0].split('.')[0]
                    if name in module_names:
                        shadow_name = name
                        break
        
        if shadow_name:
            # Check if this name is used in the function body before this import
            for j in range(func_start, i - 1):
                body_line = lines[j].rstrip()
                if body_line.lstrip().startswith('#'):
                    continue
                if body_line.lstrip().startswith('import ') or body_line.lstrip().startswith('from '):
                    continue
                # Check for name. usage (not in comments, not in strings)
                if re.search(r'\b' + re.escape(shadow_name) + r'\.', body_line):
                    rel = os.path.relpath(filepath, BASE)
                    bugs.append({
                        'file': rel,
                        'func': func_name,
                        'func_line': func_start,
                        'name': shadow_name,
                        'use_line': j + 1,
                        'import_line': i,
                        'use_text': body_line.strip()[:100],
                        'import_text': stripped[:100],
                    })
                    break
    
    return bugs

def main():
    strict = '--strict' in sys.argv
    all_bugs = []
    
    for root, dirs, files in os.walk(BASE):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]
        for f in files:
            if not f.endswith('.py'):
                continue
            fp = os.path.join(root, f)
            if '/.venv' in fp or '/venv' in fp:
                continue
            bugs = find_import_shadowing(fp)
            all_bugs.extend(bugs)
    
    if all_bugs:
        print(f"❌ IMPORT-SHADOWING BUGS FOUND: {len(all_bugs)}\n")
        for b in sorted(all_bugs, key=lambda x: (x['file'], x['func_line'])):
            print(f"  {b['file']}:{b['func_line']} {b['func']}()")
            print(f"    '{b['name']}' used at L{b['use_line']}: {b['use_text']}")
            print(f"    shadowed by import at L{b['import_line']}: {b['import_text']}")
            print()
        if strict:
            sys.exit(1)
    else:
        print("✅ No import-shadowing bugs found.")

if __name__ == '__main__':
    main()
