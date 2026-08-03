# NeuroPLC: PyTorch→IEC 61131-3 Compiler for PLCs, with a Type Theory of Certifiable Neural Architectures

> **A Type Theory of Certifiable Neural Architectures: Sharp LUT Bounds, Complexity Separation, and Compile-Aware Generalization, with an Industrial PLC Instantiation** — IEEE TNNLS submission (theory-first).

**状态：87 页 0e0w 0 undefined refs ｜ 19 定理 ｜ 2026-08-04 更新**（P3：可认证部署容量理论 Thm 17-19 加入）

Author: 刘甫悦 (板板) + Claude

---

## Quick Status（2026-08-03 全部数字经 released checkpoint 验证）

| Dimension | State |
|-----------|-------|
| Paper | **84 页，0 errors / 0 warnings / 0 undefined refs**（Thm 1–16） |
| 新定理（P1） | **T-I** 曲率感知两级最优 LUT（2.75×/3.32×）· **T-II** 编译感知 minimax 率（剪刀差+离散免费）· **T-III** necessity 第一格 · **T-IV** Besov 光滑度自适应率 |
| 界值（重算） | Δ_IA=1.38（非证书）· Δ_DA=**0.66**（1.0× 边缘，M₂^char）· M₂^max 版 5.9 —— "契约性训练是部署必需"（数据支撑） |
| WCET | **3.86 ms**（25.9× margin / 3.9%），三层估计聚类 2.86/3.2/3.86 ms（同一 Siemens 05/2024 时序基准） |
| 训练网络 | **非收缩 γ=[15.4,5.3]**（诚实报告；0.182 陈旧值已清除） |
| 实验 | E1–E20, E52–E68 + E-T1/E-T2/E-T3 + V1–V7（全部脚本+JSON） |
| 跨厂商 | backend_iec ST → **汇川 EVO810（iFA）导入成功 + FB 编译 0 语法错误**（首个非西门子证据） |
| 编译器验证 | Tier 1–3 形式证明（模板/证书/组合）+ **Tier 4 差分自测**（3,357 样本 100%/maxAE 0.47，抓过真实 emitter bug） |
| 验证边界 | 模板/证书/组合=形式证明；emitter=差分自测（非证明）；CompCert 对比已引 |
| 盲点实验 | E52：认证安全因子 **N≤12 全部 <1**，N=15 仅 1.02× 边缘（比原声称更严重，诚实报告） |
| Z3 Verification | B-spline 512/512 · ChebyKAN 496/512 · MLP 0/48 |
| CROWN comparison | DA **8.3×** tighter（重算，原 85× 为陈旧值） |

---

## 快速开始

```bash
# 1. 上手（新窗口先读）
cat RESTART_CONTEXT.md          # 交接文档（最新状态+待办）

# 2. 编译论文
cd paper && xelatex main && bibtex main && xelatex main && xelatex main
# → 84 页，0 errors，0 warnings

# 3. 定理验证（全部 PASS）
cd code
python theory/verify_sharp_constants.py             # Thm 9（锐常数 ratio 1.000000）
python theory/verify_optimal_lut.py                 # T-I（2.75×/3.32×）
python theory/verify_compile_aware_minimax.py       # T-II（剪刀差 1.003 平坦）
python theory/verify_besov_pac.py                   # T-IV（光滑度自适应）
python theory/verify_necessity_first.py             # T-III（necessity 第一格）
python theory/verify_da_bounds_recomputed.py        # 界值重算（0.66/1.38）
python -m neuroplc.differential_test                # Tier 4（3,357 样本 100%）

# 4. 双推
git push origin master && git push github master    # Gitee + GitHub
```

---

## 文档地图

| 文件 | 用途 |
|------|------|
| `RESTART_CONTEXT.md` | **主入口**：完整状态 + 发现链 + 待办 + 上手步骤 |
| `FINAL_PLAN.md` | 终极计划（P0/P1 完成，P3/P4 待） |
| `MASTER_PLAN.md` / `UPGRADE_PLAN.md` | 总纲 / 早期计划 |
| `PRE_SUBMISSION_REVIEW_2026-08-03.md` | 7-Agent 深度审稿报告（17 CRITICAL，已全清） |
| `docs/CHECKLIST.md` | P0 追踪表 |
| `docs/REVIEWER-FAQ.md` | 审稿人防御 v3（20 问） |
| `docs/cover_letter.md` | 投稿 cover letter 草稿 |
| `code/theory/verify_*.py` | 8 个定理验证脚本（全 PASS） |
| `results/theory/*.json` | 全部验证结果 |

---

## 目录结构

```
├── paper/            # LaTeX 论文（main.tex + 13 sections + figures/）
├── code/
│   ├── neuroplc/     # 编译器（frontend/backend_s7/backend_iec/differential_test/wcet）
│   ├── theory/       # 定理验证脚本（8 个 verify_*.py）
│   └── experiments/  # 74+ 实验脚本
├── results/          # 模型/SCL 输出/验证 JSON/图
├── data/             # CWRU 特征（13714×28）
├── docs/             # 审稿/计划/FAQ/追踪
├── research/         # 文献 PDF（Kratsios/Nye 等）
└── tia_project/      # TIA Portal 工程
```

*最后更新：2026-08-03 | 84页 0e0w | 16 定理 | 27 commits 双推*
