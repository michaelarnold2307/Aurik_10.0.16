"""
Convert logger.xxx(f"...{var}...") → logger.xxx("...%s...", var)
across backend/, plugins/, dsp/.

Handles:
  - Single and multiple {expr} placeholders
  - Format specs like {x:.2f} → %.2f, x
  - Escapes: {{ and }} → { and }  (literal braces)
  - Skips lines with nested f-strings or complex expressions that can't be
    safely converted (conservative fallback: leave unchanged)

Usage:
    python scripts/_fix_fstring_loggers.py [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Pattern: logger.<level>(f"...") — capture the f-string content ─────────
# Only matches single-line calls with a plain f"..." argument (no concatenation)
_LOG_CALL = re.compile(
    r"""(?P<indent>[ \t]*)(?P<logger>logger\s*\.\s*(?:info|warning|debug|error|critical|exception))\s*\(\s*f(?P<q>["'])((?:(?!(?P=q)).)*?)(?P=q)(?P<rest>[^)]*?)\s*\)""",
    re.DOTALL,
)

# ── Format-spec map: common Python format specs → printf equivalents ────────
_FMT_MAP: dict[str, str] = {
    "d": "%d",
    "i": "%d",
    "f": "%f",
    ".0f": "%.0f",
    ".1f": "%.1f",
    ".2f": "%.2f",
    ".3f": "%.3f",
    ".4f": "%.4f",
    ".5f": "%.5f",
    ".6f": "%.6f",
    "e": "%e",
    ".2e": "%.2e",
    "g": "%g",
    ".2g": "%.2g",
    "s": "%s",
    "r": "%r",
    "x": "%x",
    "X": "%X",
    "o": "%o",
    "b": "%b",
}


def _has_toplevel_comma(expr: str) -> bool:
    """Return True if expr contains a comma that is NOT inside parens or brackets."""
    depth_p = depth_b = 0
    for ch in expr:
        if ch == "(":
            depth_p += 1
        elif ch == ")":
            depth_p -= 1
        elif ch == "[":
            depth_b += 1
        elif ch == "]":
            depth_b -= 1
        elif ch == "," and depth_p == 0 and depth_b == 0:
            return True
    return False


def _extract_placeholders(fstring_body: str) -> list[tuple[str, str]] | None:
    """
    Parse the body of an f-string and extract (expression, format_spec) pairs.
    Returns None if the body is too complex to convert safely.

    Simple examples:
      "text {var}" → [("var", "")]
      "val={x:.2f}" → [("x", ".2f")]
      "{a} and {b}" → [("a", ""), ("b", "")]
    """
    placeholders: list[tuple[str, str]] = []
    i = 0
    n = len(fstring_body)
    while i < n:
        c = fstring_body[i]
        if c == "{":
            if i + 1 < n and fstring_body[i + 1] == "{":
                # Escaped brace — not a placeholder
                i += 2
                continue
            # Find matching closing brace, respecting nesting
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if fstring_body[j] == "{":
                    depth += 1
                elif fstring_body[j] == "}":
                    depth -= 1
                j += 1
            if depth != 0:
                return None  # unmatched brace — give up
            inner = fstring_body[i + 1 : j - 1]
            # Split on "!" (conversion) then ":" (format spec)
            # But only if not inside nested braces
            fmt_spec = ""
            expr = inner
            # Detect colon that is actually a format spec (not inside [] or ())
            colon_idx = _find_format_colon(inner)
            if colon_idx is not None:
                expr = inner[:colon_idx]
                fmt_spec = inner[colon_idx + 1 :]
            # Remove conversion flag (!r, !s, !a) — treat as %r / %s
            conv = ""
            if "!" in expr:
                bang = expr.rfind("!")
                conv = expr[bang + 1 :]
                expr = expr[:bang]
            expr = expr.strip()
            fmt_spec = fmt_spec.strip()
            # Safety: reject nested f-strings, lambdas, ternary operators, newlines
            if any(tok in expr for tok in ("lambda", "f'", 'f"', "\n", " if ", " else ")):
                return None
            # Reject commas that are NOT inside balanced parentheses/brackets
            # (allows .get('key', default) but rejects tuple expressions)
            if _has_toplevel_comma(expr):
                return None
            placeholders.append((expr, fmt_spec, conv))  # type: ignore[arg-type]
            i = j
        elif c == "}":
            if i + 1 < n and fstring_body[i + 1] == "}":
                i += 2
                continue
            return None  # unmatched }
        else:
            i += 1
    return placeholders  # type: ignore[return-value]


def _find_format_colon(inner: str) -> int | None:
    """Find the index of a format spec colon that is NOT inside brackets."""
    depth_p = depth_b = depth_c = 0
    for idx, ch in enumerate(inner):
        if ch in "([":
            if ch == "(":
                depth_p += 1
            else:
                depth_b += 1
        elif ch in ")]":
            if ch == ")":
                depth_p -= 1
            else:
                depth_b -= 1
        elif ch == "{":
            depth_c += 1
        elif ch == "}":
            depth_c -= 1
        elif ch == ":" and depth_p == 0 and depth_b == 0 and depth_c == 0:
            return idx
    return None


def _build_template_and_args(
    fstring_body: str,
    placeholders: list[tuple[str, str, str]],
) -> tuple[str, list[str]] | None:
    """
    Reconstruct the format string (%-style) and argument list.
    Returns None if any placeholder can't be cleanly mapped.
    """
    args: list[str] = []
    # Process placeholders in reverse so index arithmetic stays valid
    n = len(fstring_body)
    result_parts: list[str] = []
    ph_iter = iter(placeholders)
    pos = 0
    while pos < n:
        c = fstring_body[pos]
        if c == "{":
            if pos + 1 < n and fstring_body[pos + 1] == "{":
                result_parts.append("{")
                pos += 2
                continue
            # Consume placeholder
            depth = 1
            j = pos + 1
            while j < n and depth > 0:
                if fstring_body[j] == "{":
                    depth += 1
                elif fstring_body[j] == "}":
                    depth -= 1
                j += 1
            try:
                expr, fmt_spec, conv = next(ph_iter)
            except StopIteration:
                return None
            # Map format spec
            if fmt_spec:
                pct = _FMT_MAP.get(fmt_spec)
                if pct is None:
                    # Unknown format spec — fall back to %s with str(x)
                    # Only if simple numeric specs are safe
                    # Otherwise give up
                    return None
                result_parts.append(pct)
            elif conv == "r":
                result_parts.append("%r")
            else:
                result_parts.append("%s")
            args.append(expr)
            pos = j
        elif c == "}":
            if pos + 1 < n and fstring_body[pos + 1] == "}":
                result_parts.append("}")
                pos += 2
                continue
            return None
        else:
            # Escape any literal % that would interfere with printf
            if c == "%":
                result_parts.append("%%")
            else:
                result_parts.append(c)
            pos += 1
    return "".join(result_parts), args


def convert_line(line: str) -> str | None:
    """
    Convert a single-line logger f-string call.
    Returns the converted line or None if conversion is not safe.
    """
    m = re.match(
        r"""^(?P<indent>[ \t]*)(?P<call>(?:self\.)?logger\s*\.\s*(?:info|warning|debug|error|critical|exception))\s*\(\s*f(?P<q>["'])(?P<body>.*?)(?P=q)(?P<rest>[^)]*?)\s*\)(?P<tail>.*)$""",
        line,
        re.DOTALL,
    )
    if not m:
        return None
    indent = m.group("indent")
    call = m.group("call")
    m.group("q")
    body = m.group("body")
    rest = m.group("rest").strip()  # extra kwargs or trailing comma
    tail = m.group("tail").strip()  # e.g. "  # type: ignore"

    phs = _extract_placeholders(body)
    if phs is None:
        return None
    result = _build_template_and_args(body, phs)  # type: ignore[arg-type]
    if result is None:
        return None
    template, args = result

    # Reassemble call
    all_args = [f'"{template}"', *args]
    if rest:
        all_args.append(rest)
    args_str = ", ".join(all_args)
    new_line = f"{indent}{call}({args_str})"
    if tail:
        new_line += f"  {tail}"
    new_line += "\n"
    return new_line


def convert_file(path: Path, dry_run: bool = False) -> int:
    """Convert all logger f-strings in a file. Returns number of conversions."""
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        logger.warning("_fix_fstring_loggers.py::convert_file Ersatzpfad", exc_info=True)
        return 0

    lines = source.splitlines(keepends=True)
    changed = 0
    new_lines: list[str] = []
    for line_no, line in enumerate(lines, 1):
        # Quick pre-filter — handle both logger. and self.logger.
        has_logger = "logger." in line
        has_fstr = "f'" in line or 'f"' in line
        if not has_logger or not has_fstr:
            new_lines.append(line)
            continue
        converted = convert_line(line)
        if converted is not None and converted != line:
            new_lines.append(converted)
            changed += 1
        else:
            new_lines.append(line)

    if changed and not dry_run:
        new_source = "".join(new_lines)
        # Syntax guard: only write if the result parses
        import ast as _ast

        try:
            _ast.parse(new_source)
        except SyntaxError as e:
            import sys

            print(f"  SYNTAX ERROR in {path} after conversion: {e} — SKIPPED", file=sys.stderr)
            return 0
        path.write_text(new_source, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert logger f-strings to %-format")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing")
    args = parser.parse_args()

    roots = [
        Path("backend"),
        Path("plugins"),
        Path("dsp"),
        Path("denker"),
    ]
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            n = convert_file(py_file, dry_run=args.dry_run)
            if n:
                total += n
                print(f"  {'+' if not args.dry_run else '~'}{n:3d}  {py_file}")
    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{mode}Konvertiert: {total} logger f-string Aufrufe")


if __name__ == "__main__":
    main()
