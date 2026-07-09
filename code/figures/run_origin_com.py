#!/usr/bin/env python3
"""Batch run all Origin LabTalk scripts via COM (single-shot, no loops)."""
import sys, os, time
try:
    import win32com.client as win32
    o = win32.Dispatch("Origin.ApplicationSI")
    print("Origin COM connected")

    scripts = [
        "D:/neuroplc-paper/code/figures/origin_figure1_c2bv.ogs",
        "D:/neuroplc-paper/code/figures/origin_figure2_da_vs_ia.ogs",
        "D:/neuroplc-paper/code/figures/origin_figure3_models.ogs",
    ]

    for s in scripts:
        if os.path.exists(s):
            print(f"Running: {s}")
            o.Execute(f'run.file("{s}");')
            time.sleep(2)
            print(f"  Done")
        else:
            print(f"  MISSING: {s}")

    print("All Origin figures generated")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
