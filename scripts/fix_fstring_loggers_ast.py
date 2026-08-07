#!/usr/bin/env python3
r"""
fix_fstring_loggers_ast.py
===========================
Second-pass AST-based transformer for complex f-string logger calls that the
regex script (fix_fstring_loggers.py) could not handle:

    logger.info(f"val={d.get('k', 0):.2f} {name!r}")
    →  logger.info("val=%.2f %r", d.get('k', 0), name)

Uses Python's ast module (available in 3.10+) to parse each single-line call,
extract FormattedValues from JoinedStr nodes, reconstruct the %-format string
and the argument list, then rewrite the source line.

Limitations (still skipped):
- Multi-line calls (continuation \ or unbalanced parens spanning lines)
- Format specs that are themselves dynamic: f"{x:{width}.{prec}f}"
- Calls where the f-string is NOT the first argument
- Lines with syntax errors

Usage:
    python scripts/fix_fstring_loggers_ast.py [--apply] [path ...]
"""

import argparse
import ast
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_METHODS = {"debug", "info", "warning", "error", "critical", "exception"}

# Quick pre-filter: must contain logger.*( f" or f'
_QUICK_RE = re.compile(r"""logger\s*\.\s*(?:debug|info|warning|error|critical|exception)\s*\(f['"]""")


def _conversion_to_fmt(conv: int) -> str:
    """AST conversion int → % placeholder."""
    if conv == 114:  # 'r'
        return "%r"
    if conv == 115:  # 's'
        return "%s"
    if conv == 97:  # 'a'
        return "%s"  # ascii() — no direct % equivalent, use %s
    return "%s"


def _fmt_spec_str(spec_node: ast.JoinedStr | None) -> str:
    """Extract the format spec string from an ast.JoinedStr node (if simple constant)."""
    if spec_node is None:
        return ""
    # Only handle constant spec (no dynamic width/precision)
    if len(spec_node.values) == 1 and isinstance(spec_node.values[0], ast.Constant):
        return str(spec_node.values[0].value)
    return None  # type: ignore  # dynamic spec — skip this entire call


def _spec_to_percent(spec: str, conv: int) -> str | None:
    """Convert Python format spec + conversion to %-format placeholder.
    Returns None if the spec is too exotic to convert safely.
    """
    if conv in (114,):  # !r
        return "%r"
    if not spec:
        return "%s"
    # Common numeric: [sign][width][.prec]type
    m = re.fullmatch(r"""([+\- ]?)(\d*)(\.(\d+))?([dioxXeEfFgG%])""", spec)
    if m:
        sign, width, _, prec, typ = m.groups()
        s = "%"
        if sign:
            s += sign
        if width:
            s += width
        if prec is not None:
            s += "." + prec
        s += typ
        return s
    # Zero-padded integer: 05d
    m = re.fullmatch(r"""0(\d+)([dioxX])""", spec)
    if m:
        return f"%0{m.group(1)}{m.group(2)}"
    # String format: 10s, <10s, >10s
    if re.fullmatch(r"""[<>^]?\d*s""", spec):
        return "%s"
    # Boolean-like: ? — not a real Python format spec
    return None


def _transform_joinedstr(node: ast.JoinedStr) -> tuple[str, list[str]] | None:
    """
    Convert ast.JoinedStr to (format_string, [arg_source_strings]).
    Returns None if transformation is not safe.
    """
    fmt_parts: list[str] = []
    args: list[str] = []

    for part in node.values:
        if isinstance(part, ast.Constant):
            # Literal text — escape existing % signs
            literal = str(part.value).replace("%", "%%")
            fmt_parts.append(literal)
        elif isinstance(part, ast.FormattedValue):
            conv = part.conversion  # -1, 114=r, 115=s, 97=a
            spec_str = _fmt_spec_str(part.format_spec)  # type: ignore[arg-type]
            if spec_str is None:
                return None  # dynamic spec — skip

            placeholder = _spec_to_percent(spec_str, conv)
            if placeholder is None:
                return None  # exotic spec — skip

            fmt_parts.append(placeholder)
            args.append(ast.unparse(part.value))
        else:
            return None  # unexpected node type

    return "".join(fmt_parts), args


def _transform_line(line: str) -> str | None:
    """
    Transform a single source line with a logger f-string call.
    Returns the transformed line or None if not applicable / unsafe.
    """
    if not _QUICK_RE.search(line):
        return None

    stripped = line.rstrip("\n")

    # Must be self-contained on one line (balanced parens)
    open_p = stripped.count("(") - stripped.count(")")
    if open_p != 0:
        return None  # multi-line call

    # Try to parse as expression statement
    try:
        tree = ast.parse(stripped.lstrip(), mode="eval")
    except SyntaxError:
        return None

    expr = tree.body
    if not isinstance(expr, ast.Call):
        return None

    # Check it's a logger.METHOD(...) call
    func = expr.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr in _LOG_METHODS
        and isinstance(func.value, (ast.Name, ast.Attribute))
    ):
        return None

    # First argument must be a JoinedStr (f-string)
    if not expr.args or not isinstance(expr.args[0], ast.JoinedStr):
        return None

    result = _transform_joinedstr(expr.args[0])
    if result is None:
        return None

    fmt_str, new_args = result

    # Determine quote style: prefer double, avoid escaping
    q = '"'
    if '"' in fmt_str:
        if "'" not in fmt_str:
            q = "'"
        else:
            return None  # both quote types present — too complex

    # Remaining args after the f-string
    remaining_args = [ast.unparse(a) for a in expr.args[1:]]
    remaining_kwargs = [f"{kw.arg}={ast.unparse(kw.value)}" for kw in expr.keywords]
    all_args = new_args + remaining_args + remaining_kwargs

    # Reconstruct the logger accessor (preserve how it's accessed: self.logger, logger, etc.)
    logger_accessor = ast.unparse(func.value)
    method_name = func.attr

    if all_args:
        args_str = ", ".join(all_args)
        new_call = f"{logger_accessor}.{method_name}({q}{fmt_str}{q}, {args_str})"
    else:
        new_call = f"{logger_accessor}.{method_name}({q}{fmt_str}{q})"

    # Validate: new call must also parse without errors
    try:
        ast.parse(new_call, mode="eval")
    except SyntaxError:
        return None

    # Preserve original indentation
    indent = len(line) - len(line.lstrip())
    return line[:indent] + new_call + "\n"


def _process_file(path: Path, apply: bool) -> int:
    try:
        original = path.read_text(encoding="utf-8")
    except Exception:
        logger.warning("fix_fstring_loggers_ast.py::_verarbeiten_file Ersatzpfad", exc_info=True)
        return 0

    lines = original.splitlines(keepends=True)
    changed = 0
    new_lines: list[str] = []

    for line in lines:
        transformed = _transform_line(line)
        if transformed is not None and transformed != line:
            new_lines.append(transformed)
            changed += 1
        else:
            new_lines.append(line)

    if changed:
        # Final syntax check on full file
        try:
            ast.parse("".join(new_lines))
        except SyntaxError:
            return 0  # Don't write if file-level syntax broken
        if apply:
            path.write_text("".join(new_lines), encoding="utf-8")

    return changed


def _collect_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
        elif root.is_dir():
            for p in sorted(root.rglob("*.py")):
                if "__pycache__" not in p.parts:
                    files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["backend", "dsp", "plugins", "denker"],
    )
    args = parser.parse_args(argv)

    base = Path(__file__).parent.parent
    roots = [base / p for p in args.paths]
    files = _collect_files(roots)

    total_files = 0
    total_lines = 0

    for f in files:
        n = _process_file(f, apply=args.apply)
        if n:
            total_files += 1
            total_lines += n
            rel = f.relative_to(base)
            print(f"  {'CHANGED' if args.apply else 'WOULD CHANGE'} {rel}: {n} lines")

    action = "changed" if args.apply else "would change"
    print(f"\nTotal: {total_lines} lines {action} in {total_files} files.")
    if not args.apply:
        print("Run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
