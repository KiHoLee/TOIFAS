"""Run the stages that live outside exp_full, after the main sweep.

Order matters: exp_refresh trains its own model around the same base keys,
exp_kpa and exp_permkpa attack the main configuration, and exp_real_sec
reuses the main configuration on real token streams. Each writes only CSV.
"""
import runpy
import sys
import time

STAGES = [
    ("known-plaintext attack", "exp_kpa.py"),
    ("permutation known-plaintext attack", "exp_permkpa.py"),
    ("key-refresh layer", "exp_refresh.py"),
    ("real token streams", "exp_real_sec.py"),
]

for label, script in STAGES:
    print(f"\n{'=' * 60}\n== {label}  ({script})\n{'=' * 60}", flush=True)
    t0 = time.time()
    try:
        runpy.run_path(script, run_name="__main__")
    except Exception as exc:                       # keep going, report at end
        print(f"[FAIL] {script}: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
    print(f"[done] {label} in {time.time() - t0:.0f} s", flush=True)

print("\nall remaining stages complete")
