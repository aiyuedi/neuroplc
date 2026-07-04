"""
NeuroPLC — From PyTorch to PLC: Automated Code Generation for Deploying
          Intelligent Fault Diagnosis on IEC 61131-3 Controllers.

Package modules:
    utils/       — MLflow tracking, config helpers
    compiler.py  — PyTorch model → IEC 61131-3 SCL compiler (Phase 2)
    scl_templates.py — SCL code templates (Phase 2)
    validator.py — SCL output validation (Phase 2)
"""

__version__ = "2.0.0"  # v2: KAN + VRM-KD upgrade
__author__ = "Fuyue Liu (刘甫悦)"
