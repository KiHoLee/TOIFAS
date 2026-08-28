# -*- coding: utf-8 -*-
"""Regenerate every learned-key artifact under the regularized loss.

learned_model now trains under the two penalties of Section V-C, so
every *_learned file has to be rebuilt from it. sens runs before brute
because brute re-reads the sensitivity sweep it produced.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_learned as E
import exp_full as F

# stage_N is Fig. 2's learned SNR sweep and lives in exp_full, so it is
# named here rather than in exp_learned's own list
STAGES = [F.stage_N, E.keylen, E.jamming, E.kpa, E.refresh, E.compare,
          E.sens, E.brute, E.real]

if __name__ == "__main__":
    only = sys.argv[1:]
    for fn in STAGES:
        if only and fn.__name__ not in only:
            continue
        print("=" * 60)
        print("stage", fn.__name__, flush=True)
        fn()
    print("[done] every learned artifact rebuilt")
