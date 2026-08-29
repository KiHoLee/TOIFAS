# -*- coding: utf-8 -*-
"""Verify every bibliography entry against the article's own first page.

The standard asks for one verdict per entry against the publisher record
(12.23), and for a missing issue number to be completed "when the
publisher record shows one" (8.6). Both are answerable from ref/,
because the stored PDF is the published article and its running head
carries the volume, the issue when the journal has one, the year and the
first page.

This exists because an audit read eight entries with no `number` field
as incomplete. They are not: IEEE now publishes TIFS, JSAC, TWC, TCOM
and TSP with continuous volume pagination, and those articles' running
heads read "VOL. n, YEAR" with no issue at all. Adding a number there
would invent data. The check makes the distinction mechanical so the
finding is not raised again.

Entries with no stored PDF are reported as unverifiable rather than
passed, so the count of what remains unchecked is visible.

Run: python code/check_bib_records.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "references.bib"
REF = ROOT / "ref"

# a running head, in the several shapes the venues use
HEADS = [
    # IEEE journal with an issue: VOL. 22, NO. 12,  and VOL. IT-24, NO. 3,
    re.compile(r"VOL\.\s*(?:[A-Z]{2}-)?(\d+)\s*,\s*NO\.\s*(\d+)", re.I),
    # IEEE journal on continuous volume pagination: VOL. 20, 2025
    re.compile(r"VOL\.\s*(\d+)\s*,\s*(?:19|20)\d{2}", re.I),
    # a journal that prints volume(issue): 24(6):801-812
    re.compile(r"\b(\d+)\((\d+)\)\s*:"),
]
def entries():
    txt = BIB.read_text(encoding="utf-8")
    for m in re.finditer(r"@(\w+)\s*\{([^,]+),(.*?)\n\}", txt, re.S):
        yield m.group(2).strip(), m.group(1).lower(), m.group(3)


def field(body, name):
    m = re.search(r"\b%s\s*=\s*\{([^}]*)\}" % name, body)
    return m.group(1).strip() if m else None


def head_of(pdf):
    import fitz
    d = fitz.open(pdf)
    t = " ".join(d[0].get_text().split())
    d.close()
    return t


def main() -> int:
    checked = ok = 0
    problems, unverifiable, noissue = [], [], []
    for key, kind, body in entries():
        if key == "BSTcontrol":
            continue
        pdf = REF / (key + ".pdf")
        if not pdf.exists():
            unverifiable.append(key)
            continue
        checked += 1
        head = head_of(pdf)
        vol, num = field(body, "volume"), field(body, "number")
        why = []

        printed_vol = printed_num = None
        for rx in HEADS:
            m = rx.search(head)
            if m:
                printed_vol = m.group(1)
                printed_num = m.group(2) if m.lastindex and m.lastindex > 1 \
                    else None
                break
        if printed_vol and vol and printed_vol != vol:
            why.append("volume %s printed, %s in bib" % (printed_vol, vol))
        if printed_num and num and printed_num != num.split("--")[0]:
            why.append("issue %s printed, %s in bib" % (printed_num, num))
        if printed_num and not num:
            why.append("issue %s printed, none in bib" % printed_num)
        # a volume with no printed issue is the continuous-pagination case
        # 8.6 exempts, and a scanned cover page that omits an issue the
        # entry carries is not evidence against the entry
        if not printed_num:
            noissue.append(key)

        pages = field(body, "pages")
        if pages:
            first = pages.split("--")[0].strip()
            if first and first not in head.replace(",", ""):
                why.append("first page %s not on the printed page" % first)

        year = field(body, "year")
        if year and year not in head:
            why.append("year %s not on the printed page" % year)

        if why:
            problems.append((key, why))
        else:
            ok += 1

    for key, why in problems:
        print("  MISMATCH  %-26s %s" % (key, "; ".join(why)))
    print()
    print("verified against ref/: %d of %d entries, %d clean, %d mismatched"
          % (checked, checked + len(unverifiable), ok, len(problems)))
    if noissue:
        print("printed record carries no issue number (%d): %s"
              % (len(noissue), ", ".join(sorted(noissue))))
    if unverifiable:
        print("no stored PDF, not verifiable here: %s"
              % ", ".join(sorted(unverifiable)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
