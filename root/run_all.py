"""
DS-3002 Assignment #4 — Master Runner
Run this single file to execute all parts in order.
Usage: python run_all.py
"""

import os, sys, subprocess

os.makedirs('outputs', exist_ok=True)
os.makedirs('models',  exist_ok=True)

SCRIPTS = [
    ('Preprocessing',    'notebooks/preprocessing.py'),
    ('Part A',           'notebooks/part_a_unsupervised.py'),
    ('Part B',           'notebooks/part_b_ensemble.py'),
    ('Part C',           'notebooks/part_c_ann.py'),
    ('Part D',           'notebooks/part_d_cnn.py'),
]

for label, script in SCRIPTS:
    print(f"\n{'='*65}")
    print(f"  RUNNING: {label}  ({script})")
    print(f"{'='*65}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] {script} exited with code {result.returncode}")
        sys.exit(result.returncode)

print("\n" + "="*65)
print("  ALL PARTS COMPLETE")
print("  Plots   → outputs/")
print("  Models  → models/")
print("  Dashboard: streamlit run app.py")
print("="*65)
