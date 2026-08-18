# -*- coding: utf-8 -*-
"""Guard against mangled TeX control sequences.

Shell heredocs silently turn a backslash escape into the control
character it names, so \times becomes a tab followed by "imes" and \ref
becomes a carriage return followed by "ef". LaTeX compiles both without
an error and prints the wreckage, so the build log cannot catch this.
This scan can.
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
    for h in hits:
        print("  FAIL  " + h)
    if hits:
        print("tex health: %d suspicious sequences" % len(hits))
        return 1
    print("  PASS  tex health :: no mangled control sequences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
