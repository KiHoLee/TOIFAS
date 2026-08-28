# -*- coding: utf-8 -*-
"""Rerun the four measurements the audit found mis-specified.

refresh   the learned rows ran 8 blocks while the table says 24
jamming   the learned curve ran a 5 dB grid inside a 2 dB figure
compare   the learned row averaged four users inside a user-1 table
enum      the enumeration attacks ran on the unpenalized learned keys
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_learned as E
import check_family_enum as F

if __name__ == "__main__":
    for fn in (E.refresh, E.jamming, E.compare, F.run):
        print("=" * 60)
        print("stage", fn.__module__ + "." + fn.__name__, flush=True)
        fn()
    print("[done] audit reruns complete")
