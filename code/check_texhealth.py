# -*- coding: utf-8 -*-
"""Guard against mangled TeX control sequences.

Shell heredocs silently turn a backslash escape into the control
character it names, so \times becomes a tab followed by "imes" and \ref
becomes a carriage return followed by "ef". LaTeX compiles both without
an error and prints the wreckage, so the build log cannot catch this.

A second failure mode has the same property. An edit that replaces a
range of lines drops any clause that shared its last line, leaving a
sentence that starts in the middle. That also compiles and prints. Both
scans are here.
"""
import re
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parents[1] / "main.tex"
CTRL = {"\t": "TAB", "\r": "CR", "\x08": "BS", "\x0c": "FF",
        "\x07": "BEL", "\x0b": "VT", "\x00": "NUL"}
# a macro name shorn of its first letter, which is what the escape ate
STUBS = ["ef{", "abel{", "ite{", "extbf{", "extit{", "ext{", "imes",
         "rac{", "eft(", "ight)", "ho_", "elta", "psilon", "ambda",
         "igma", "ewline", "otag", "uad", "nderline", "ag{", "egin{",
         "nd{", "aption{", "ilde{", "ar{", "at{", "ec{"]
PAT = re.compile("(?<![" + chr(92)*2 + "A-Za-z0-9])(" +
                 "|".join(re.escape(x) for x in STUBS) + ")")


ABBREV = ("e.g.", "i.e.", "al.", "Eq.", "Fig.", "Sec.", "vs.", "cf.",
          "resp.", "etc.")


def truncated(lines):
    """A period ending a line followed by a lowercase line start is the
    signature of a lost sentence head."""
    out = []
    for i in range(1, len(lines)):
        a, b = lines[i - 1].rstrip(), lines[i]
        if not a.endswith(".") or a.endswith(ABBREV):
            continue
        if b[:1].islower() and b[:1].isalpha():
            out.append((i + 1, a[-42:], b[:44]))
    return out


def midline_comment(lines):
    """A "%" with text after it on the same line comments that text out.
    At the end of a line it is a deliberate continuation, and escaped as
    "\\%" it is a literal percent sign, so only the middle case is a bug."""
    out = []
    for n, line in enumerate(lines, 1):
        i = 0
        while True:
            i = line.find("%", i)
            if i < 0:
                break
            if i and line[i - 1] == chr(92):
                i += 1
                continue
            rest = line[i + 1:]
            if rest.strip():
                out.append((n, line[max(0, i - 40):i + 40]))
            break
    return out


def main():
    if not TEX.exists():
        print("  SKIP  tex health :: main.tex not in this package")
        return 0
    lines = TEX.read_text(encoding="utf-8").split("\n")
    hits = []
    for i, line in enumerate(lines, 1):
        for ch, name in CTRL.items():
            if ch in line:
                hits.append("CTRL %s line %d: %r" % (name, i, line[:90]))
        for m in PAT.finditer(line):
            seg = line[max(0, m.start() - 30):m.start() + 30]
            hits.append("STUB %r line %d: %r" % (m.group(1), i, seg))
    for n, seg in midline_comment(lines):
        hits.append("PCT line %d: %s" % (n, seg))
    cut = truncated(lines)
    for ln, a, b in cut:
        hits.append("CUT line %d: ...%s || %s" % (ln, a, b))
    for h in hits:
        print("  FAIL  " + h)
    if hits:
        print("tex health: %d suspicious sequences" % len(hits))
        return 1
    print("  PASS  tex health :: no mangled sequences, comments, or truncated sentences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
