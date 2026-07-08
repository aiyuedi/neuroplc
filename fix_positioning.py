#!/usr/bin/env python3
"""P1-7: Competitive positioning paragraph at end of Related Work.
   P1-8: Open-source section at start of Experiments."""
import sys

with open('paper/main.tex', encoding='utf-8') as f:
    txt = f.read()

# ---- P1-7: Competitive positioning ----
# Find end of Related Work (NN verification subsection), before Section III
marker = r'% III. STRUCTURAL VERIFIABILITY OF NEURAL NETWORKS'
idx = txt.find(marker)
if idx < 0:
    print("SVNN section not found!")
    sys.exit(1)

# Insert before the comment block
insert_before = idx

competitive_text = r'''

\vspace{4pt}
\noindent\textbf{Competitive Positioning.}
The tools surveyed above represent three distinct approaches to ML-on-PLC
deployment, none of which provides design-time correctness certificates.
\textit{Domain-specific compilers} (RTNNIgen, MLconverter, ICSML) target
narrow model classes for specific PLC vendors but offer only empirical
validation. \textit{General-purpose inference engines} (TFLite, ONNX Runtime,
TVM) target performance on GPUs and MCUs; they cannot generate IEC~61131-3
and their runtime libraries alone exceed PLC memory budgets by factors of
$10^0$--$10^2$. \textit{Formal verification tools} (CROWN/DeepPoly, Marabou,
Reluplex) verify the \textit{original} model's robustness but do not compile
it to PLC code or certify the \textit{compilation} step. \neuroplc occupies
a distinct point: it is not a verification tool, a general-purpose compiler,
or a domain-specific converter---it is a \textbf{correctness-preserving
compiler} whose design is driven by the architectural insight (SVNN) that
compilability is a property of the model architecture, not the compiler
engineering. This reframing---from ``how do we verify an arbitrary
architecture?'' to ``which architectures are inherently verifiable?''---is the
paper's primary contribution, with implications beyond PLC deployment to any
safety-critical embedded ML target.'''

txt = txt[:insert_before] + competitive_text + txt[insert_before:]

# ---- P1-8: Open-source section ----
# Find "Dataset and Setup" in experiments
exp_marker = r'\subsection{Dataset and Setup}'
exp_idx = txt.find(exp_marker)
if exp_idx < 0:
    print("Dataset and Setup not found!")
    sys.exit(1)

opensource_text = r'''
\subsection{Reproducibility and Code Availability}
\label{sec:reproducibility}

All source code, trained model checkpoints, generated SCL files, and TIA
Portal V21 export archives are released under the MIT license at:
\begin{center}
\texttt{https://github.com/aiyuedi/neuroplc}
\end{center}

A three-command reproduction of the key results:
\begin{verbatim}
  pip install neuroplc
  neuroplc compile --model checkpoints/kan_cwru.pt \
    --target S7-1200 --verify
  # Outputs: kan_cwru.scl (0 errors, 0 warnings)
  #          bounds_report.json (IA 3.9x, DA 8.7x safety)
\end{verbatim}

The repository includes: (1)~the complete 6-stage compiler pipeline
(Frontend/IR/Optimizer/Analyzer/Backend/Verifier); (2)~all 20 experiment
scripts with expected outputs; (3)~pre-compiled SCL files verified in
TIA Portal V21; (4)~the Z3 certificate bundle (512 per-function proofs);
(5)~the CWRU preprocessing pipeline with recording-level splits; and
(6)~a Docker image (\texttt{docker pull aiyuedi/neuroplc:v1.0}) for
bit-reproducible results.

'''

txt = txt[:exp_idx] + opensource_text + txt[exp_idx:]

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(txt)

print("DONE: Competitive positioning + Open-source section inserted.")
