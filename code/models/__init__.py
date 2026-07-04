"""
NeuroPLC — Model definitions.

Models:
    TeacherCNN     — 1D-CNN + Self-Attention, ~50K params, input: (1, 1024)
    StudentKAN     — Shallow KAN, ~300 params, input: (28,)
    StudentMLP     — MLP baseline, ~1600 params, input: (28,)
"""

from .teacher_cnn import TeacherCNN
from .student_kan import StudentKAN, KANLinear
from .student_mlp import StudentMLP

__all__ = ["TeacherCNN", "StudentKAN", "KANLinear", "StudentMLP"]
