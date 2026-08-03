# NeuroPLC — 新窗口上手文档（2026-08-04 v2）

> **若在新 session 中接手本项目，先读此文件，再执行「上手第一步」。**
> v2 由 2026-08-04 单日全会话更新：P3 容量理论 + P4 审稿修复 + 优化轮全部完成。

---

## 1. 项目定位（当前）

**论文标题**：A Type Theory of Certifiable Neural Architectures: Sharp LUT Bounds, Complexity Separation, and Compile-Aware Generalization, with an Industrial PLC Instantiation

**目标刊**：IEEE TNNLS（理论为主，PLC为工业案例）

**一句话**：第一个证明"哪些神经网络架构天生可认证、以什么锐常数、什么复杂度、什么验证成本"的容量理论框架——universal packing 线 + 三层验证分层 + 架构即码选择律。

**当前状态**：**89页，0e0w，0 undefined refs，19 定理（Thm 17-19 = 可认证部署容量理论）**，16 commits 双推

---

## 2. 2026-08-04 全会话成果（一次完成）

### P3 容量理论（金字塔重组，commit 5d474ba）
- **`paper/section_capacity.tex`**：Thm 17 分层（universal packing 线 c_ent·M_k·N^{-k} + 三层：闭式界家族 c_k / kink 率退化 / 结构最优 c_* conjecture 指数验证 + CP 内 necessity 4 步证明）、Thm 18 权衡（常数购买 gap≥1.53× / 类适配免费 / HLZ 例外）、Thm 19 架构选择律（Lambert-W 闭式交叉 eq:crossover-closed）+ CP 证明系统（PCC+Gappa/Sollya/LeanCert 血统）+ related work 边界段
- Abstract/Intro/Conclusion 金字塔叙事；references.bib +24 条（全部 lit-scout 查证）
- 编号链：Thm 17/18/19（setcounter 从 16 起，1-16 零扰动）

### P4 审稿闭环（6-Agent + 修复，commit edb9b3c/fdaaf06）
- **`PRE_SUBMISSION_REVIEW_2026-08-04.md`**：6-Agent 合并报告（A6 判 Significant/Incremental + Revise；A2 数字全对 0 CRITICAL；A3 4 MAJOR 精度问题；A4 4 MAJOR 数学错误；A5 4 表陈旧；A1 4 CRITICAL 散文）
- 数学核心修复：节点参数化 N（修复 B 符号冲突）、stratum-1 常数 = c_k（撤销 c_fix 投影声称——A4 证明 LS 是仿射不能低于插值）、necessity 前提可计算性假设显式 + tight 限定 + Condition 1 撤回、类级归约降级 conjecture（红线③）、PCP 句移除（建模约定）、C(N,V) 形式定义、Regime 4 归位
- 数字诚实化：4.4×→25.9×、E41 架构统一（代码裁决 [28,16,4] 0/16）、E56 重算 1.90、E57 8.3×/5.45 撤回（口径不统一）、Seg-DA 11.9× 撤回、MNIST 声明修正、first verified→self-verifying
- 图表：15 图全引用（5 处补引用）、fig17 (V_B,ε) 平面图新增、figure_list v2、3 个孤立图删除、tab:cert_thresholds 重算（IA 需 N≥22）
- 投稿材料：REVIEWER-FAQ v4（Q21-26 容量防御）、cover letter v2、figure_list v2、author bio

### 优化轮（commit 9e3c1c9/f01c775）
- **理论**：c_ent ≈ 0.079（构造性下界，verify_packing_constant.py）、c_* ≲ 0.082 vs c_k=0.125（verify_width_constant.py，gap≥1.53×）、ρ*(B) Lambert-W 闭式、比特参数化 local-vs-global code gap remark
- **数据**：E56 3L DA 重算 1.90（0.98 也是陈旧）、E57 机制确认（CROWN=IBP for KAN）、口径统一策略（E68 权威源）
- **训练**：**E-T9 软收缩**（train_contractive_kan.py 四轮）：γ [15.4,5.3]→[1.03,1.38]（11×/3.8×），98.5% 精度——权衡曲线量化（γ 越紧精度越低）；全收缩（γ<1）受 B-spline 基幅值≥1 架构常数限制——诚实边界明确化 + 未来工作（有界幅值基）

---

## 3. 关键文件索引

| 文件 | 用途 |
|------|------|
| `paper/main.tex` | 主文件（89页，0e0w，Thm 1-19） |
| `paper/section_capacity.tex` | **容量理论（2026-08-04）**：Thm 17-19 + CP + 常数 remark + related work 边界 |
| `paper/section_svnn.tex` | SVNN 理论（input svnn_theorems/lut_sharp/trichotomy/capacity） |
| `paper/section_svnn_theorems.tex` | Thm 5p/6p/14/15 + Box延续 + **E-T9 软收缩段** |
| `paper/section_lut_sharp.tex` | Thm 9p/13（锐常数 + 最优 LUT） |
| `paper/section_trichotomy.tex` | Thm 10p/16（三分法 + necessity 第一格） |
| `PRE_SUBMISSION_REVIEW_2026-08-04.md` | **6-Agent 审稿合并报告**（修复清单来源） |
| `docs/CAPACITY_DESIGN.md` | 容量理论设计文档 v4（三定理 + necessity 链 + 三条红线） |
| `docs/REVIEWER-FAQ.md` | 审稿人防御 v4（Q1-26） |
| `docs/cover_letter.md` | 投稿信 v2（容量主线） |
| `docs/figure_list.md` | 图清单 v2（15 图真实清单 + 3 孤立已删） |
| `docs/CHECKLIST.md` | P0 追踪表 |

**验证脚本速查**（全部 PASS）：
```bash
cd code
python theory/verify_sharp_constants.py       # Thm 9p（锐常数 1.000000）
python theory/verify_optimal_lut.py           # T-I（2.75×/3.32×）
python theory/verify_compile_aware_minimax.py # T-II
python theory/verify_besov_pac.py             # T-IV
python theory/verify_necessity_first.py       # T-III
python theory/verify_da_bounds_recomputed.py  # 界值重算（0.66/1.38）
python theory/verify_capacity.py              # E-T4：packing 常数匹配（KT 1.000000/排序）
python theory/verify_stratification.py        # E-T5：三层分离（-1.11 vs -2.05）
python theory/verify_decision_law.py          # E-T6：决策律相变（exp_slope=-ρ）
python theory/verify_packing_constant.py      # E-T7：c_ent≈0.079 构造下界
python theory/verify_width_constant.py        # E-T8：c_*≲0.082（gap≥1.53×）
python theory/make_capacity_figures.py        # fig17 (V_B,ε) 平面图
python experiments/train_contractive_kan.py   # E-T9：软收缩训练（γ→1.03/1.38）
cd paper && xelatex main && bibtex main && xelatex main && xelatex main  # 编译
```

---

## 4. 剩余待办

### 投稿前（高优先）
1. **TNNLS 页数策略（板板决策）**：89 页单栏 ≈ 44 页双栏 vs ~14 页限制——压缩核心理论+实验转补充材料 vs 分流（IMA J. Inf. Inference / FoCM 理论侧；IEEE TII 系统侧）
2. **FIXME/TODO 源码注释清扫**（~30 处——不影响 PDF，最终提交前一次 grep 清扫）
3. **tab:summary 加 E-T 行**（E-T1-E-T9 实验入汇总表——计数文字已更新）
4. **Box-Continuation 域加宽实验（C11）**：加宽层 2 LUT 域 → 重编译 512/512（依赖编译工具链）
5. **T1 归约尝试**：stratum-3 类级归约（ε-Verify → 结构最优性）——失败保持 conjecture
6. **定理顺序重编号**（A4：文档顺序乱 + Thm 12 缺失）——高风险，建议搁置

### 理论深化（中优先）
7. 有界幅值 B-spline 基 → 全收缩训练（γ<1）——E-T9 标注的未来工作
8. 三分法 necessity 绝对形式（CP 内定理 + conjecture 坐标化——保持开放问题定位）

---

## 5. 三个诚实底线（勿动）

1. **训练网络非收缩（γ=[15.4,5.3]）**——E-T9 软收缩演示（γ→[1.03,1.38]，11× 改善）已诚实写入；全收缩受 B-spline 基幅值限制（架构常数，未来工作）
2. **三分法 necessity 全篇 conjecture 标注**——CP 内相对形式是定理；绝对形式（P≠NP 级）是开放问题——这是贡献不是缺陷
3. **手写编号脆弱**——插新定理必须手动 bump 下游计数器（setcounter 链，本次踩坑 2 次）

---

## 6. 上手第一步（新session立刻执行）

```
1. 读本文件（你正在读）
2. cd D:/neuroplc-paper && git log --oneline -5   # 确认 HEAD（16 commits 双推）
3. 编译验证：cd paper && xelatex main && bibtex main && xelatex main && xelatex main
   → 期望：89页，0 errors，0 undefined refs
4. 读 PRE_SUBMISSION_REVIEW_2026-08-04.md（审稿状态）与 docs/CAPACITY_DESIGN.md（理论设计）
5. 按「剩余待办」继续：先定页数策略 → FIXME 清扫 → 投稿
```

---

## 7. 环境备忘

- Python 3.14.3（系统）+ `D:\dev-tools\research\venv\`（含 torch/matplotlib/scipy/sklearn）；模型路由 V4-Flash
- 论文编译：MiKTeX xelatex（`D:\miktex\`）
- 模型：`results/student/kan_kd_vrmKD_best.pt`（主，99.93%）；`kan_contractive.pt`（E-T9 软收缩，98.5%）
- 数据：`data/processed/features_X.npy`（CWRU 28-D，13714 样本）
- Git：origin=Gitee, github=GitHub（双推工作流）

*最后更新：2026-08-04 | 板板 + Claude | 89页 0e0w | 19 定理 | 16 commits 双推 | 容量理论 + 审稿闭环 + 优化轮完成*
