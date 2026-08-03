# NeuroPLC 终极计划 FINAL PLAN：开创级（最高级别）+ 奠定级

> 生效日：2026-08-03 | 上游：MASTER_PLAN.md（总纲）+ UPGRADE_PLAN.md（早期版，本文件取代其 Phase 2 部分）
> 统一主题：**"离散化执行的最优性理论"**——在给定存储预算下，LUT 执行的每一环（网格、密度、架构类、学习率）都最优，且证明不可能更好。可认证架构 = 处处最优的架构。

---

## 0. 验收定义（最高级别 = 什么）

### 开创级（最高级别，2027-03 投稿前达成）
- [ ] **T-I 最优恢复定理**：LUT 表格约束下的最优恢复常数 + 非均匀最优网格定理 + 与均匀网格的显式 gap 常数（证明 + 数值极值族 + 可部署算法）
- [ ] **T-II 编译感知 minimax 定理**：部署风险下界，N*≍n^{1/(2k)} 证明为 minimax 最优编译策略（含下界证明，非 heuristic）
- [ ] **T-III 必要性第一格**：在"design-time-computable affine 证书"类下证明三分法 necessity（conjecture → 定理，至少非平凡子类）
- [ ] **T-IV Besov 编译感知 PAC**：维度无关样本复杂度，与 Kratsios 2025 最优率对接
- [ ] 论文结构重构为理论主线 + 全部审稿 CRITICAL 清零（CHECKLIST.md 全绿）
- [ ] 双厂商实测（iFA 编译 backend_iec ST + 仿真下载）写入论文
- [ ] review-paper 全量审稿 0 CRITICAL → 投 TNNLS

### 奠定级（2028-06 毕业前达成）
- [ ] 复杂度谱系全图（全部常见激活类分类定理）
- [ ] Coq 机械化（Galois 修复版 + T-I 最优网格）
- [ ] 收缩训练端到端（γ<1, ≥99%）+ 物理 PLC 实测
- [ ] VNN-COMP 2028 新赛道提案（LUT 编译验证基准）

---

## 1. 论文新结构（目标目录树）

```
main.tex
├── I. Introduction（重写：从"我们编译 KAN"→"哪类网络可认证且处处最优"）
├── II. Preliminaries（新增：符号表 + 术语统一——γ/margin/M₂/h 全部单义）
├── III. Type System（IR 类型规则——保留，重命名为"可认证性判定规则"）
├── IV. 最优恢复理论（新，T-I：LUT 最优恢复 + 最优网格）← section_lut_sharp.tex 重写
├── V. 验证复杂度谱系（T-III necessity + Thm 5 ε-分离 + Thm A）← 三分法重写
├── VI. 编译感知学习理论（T-II minimax + T-IV Besov PAC + Thm 6 升级）← svnn_theorems 重写
├── VII. 编译器与验证（4-tier + 算法4 + 双厂商后端）
├── VIII. 工业案例（CWRU + 最优网格实证 + 跨厂商实测）
├── IX. 实验（E 编号重排：新定理各配 E 系列）
└── X. 结论（necessity 开放问题 = 领域公开问题）
```

**删除/弱化**：folklore 叙事 → "显式窗口积常数 + 认证极值族"；15 个"first"→ 保留 ≤3 个并全部 "to our knowledge"；RBF-KAN/MNIST 声称 → 改为"理论性/计划中"。

---

## 2. 四大定理规格（每个定理 = 1 节论文 + 1 个验证脚本 + 1 个实验）

### T-I：LUT 最优恢复定理 ★ 最优先（2026-08-17 → 09-30）
**目的**：把 Thm 9(3)（审稿 CRITICAL：无证明断言）变成真定理；把"教科书重述"变成新问题。
**半形式化陈述**（精确版待读 MRW 1974 后定稿）：
> 设 f ∈ C^k，LUT 表由节点集 X_N = {x_i} 与节点值 {f(x_i)} 组成，恢复算法族 A = {所有仅依赖节点值的算法}（分段常数/线性插值等）。则：
> (a) minimax 恢复常数 R*_N(k) = sup_f min_A ‖f − A(f)‖ / ‖f^(k)‖∞ 等于推广的 perfect-spline 范数；
> (b) 对 LUT 约束（逐节点存储、固定预算 N），**均匀网格在 k≥2 时严格次优**：gap 常数 g_k > 0 显式给出；
> (c) 最优网格由 Chebyshev perfect spline 零点确定（de Boor optknt 的 LUT 版算法，O(N) 可部署）。
**证明策略**：Micchelli-Rivlin-Winograd 对偶 + LUT 约束下的离散化处理；极值族 = 截断 perfect spline。
**验证脚本**：`code/theory/verify_optimal_lut.py` → `results/theory/ti_optimal_lut.json`（gap 常数数值确认、最优网格 vs 均匀网格误差比、随机函数族统计验证）。
**实验 E-T1**：CWRU 上同精度下 LUT 预算减半（最优网格 vs 均匀）→ 工程卖点。
**风险闸**：先读 MRW 1974（opencite 取原文），若 LUT 约束下无新常数 → 降级为"均匀网格 gap 的显式化 + 算法化"，仍支撑 Thm 9(3) 的证明完备化。

### T-II：编译感知 minimax 定理（2026-10-01 → 11-15）
**目的**：回应"分辨率匹配 = 经典带宽律重述"（审稿 CRITICAL）；把 heuristic 变定理。
**半形式化陈述**：
> 对学习算法 L、编译方案 C（将 f̂ 映射到 N 点 LUT 表，预算 |N| = B），定义部署风险 R_deploy = 𝔼[R(f̂, f*) + bias_C(f̂)]。则：
> (a) minimax 下界：inf 编译方案 sup 分布 R_deploy ≥ c·n^{−k/(2k+1)}（对 k-光滑目标类）；
> (b) NeuroPLC 的均匀网格 + N*≍n^{1/(2k)} 达到该率（上界匹配）；
> (c) 因此 N*≍n^{1/(2k)} 是 **minimax 最优编译策略**——任何编译器不能以更少内存达到同等部署风险。
**证明策略**：统计 minimax（Fano/插值下界）+ T-I 的恢复常数；上界用现有 Thm 6 分解。
**验证脚本**：`code/theory/verify_compile_aware_minimax.py` → `results/theory/tii_minimax.json`（合成目标类上的下界数值行为）。
**实验 E-T2**：合成类（已知光滑度）上扫 n × N，确认率匹配 n^{−k/(2k+1)}。

### T-III：必要性第一格（2026-11-16 → 12-31，最难，可滚动）
**目的**：三分法 necessity 从 conjecture 升级为有条件定理（审稿 CRITICAL + Agent 6 明确点名这是"单点最大升级"）。
**半形式化陈述**：
> 在"design-time-computable affine 证书"类 C（证书 = 编译期可从模型参数计算的仿射界）下：
> 若架构类 A 满足 (P2) 锐常数（T-I 意义下达到 minimax 恢复率），则 A 的验证复杂度多项式 ⇔ A ⊆ SVNN ∪ (有限例外)。
**证明策略**：反证——若 A ∉ SVNN 且达到锐常数，构造耦合实例（Thm 5 产品门家族）使验证 NP-hard；或证明非 SVNN 类在 C 类证书下无法达到锐常数（用 T-I 的最优恢复下界：均匀/可计算网格 vs 最优网格的 gap 不可消除）。
**验证脚本**：`code/theory/verify_necessity.py`（构造性：非 SVNN 类的最优恢复 gap 数值验证 + 复杂度归约检查）。
**风险闸**：若完整失败 → 在论文中保留为"形式化 conjecture + 三面证据（T-I gap、Thm 5 归约、谱系表）"，并将"necessity 开放问题"明确为领域公开问题（本身可成为引用点）。

### T-IV：Besov 编译感知 PAC（2026-10-15 → 12-15，与 T-II/T-III 并行）
**目的**：与 Kratsios 2025（Res-KAN Besov 最优率）对话，形成维度无关样本复杂度（审稿"经典律重述"批评的终极回应）。
**半形式化陈述**：
> 目标函数类 B^s_{p,q}(X)（Besov），KAN 逼近率（Kratsios）+ LUT 编译 bias（T-I 常数）→
> R_deploy(n, N) ≤ c₁·n^{−2s/(2s+d)} + c₂·N^{−2k}（维度项只在第一项）
> 且存在编译策略使 N(n) ≍ n^{1/(2k)} 时达到总率 n^{−min(2s/(2s+d), 1)} 阶。
**证明策略**：Kratsios 的伪维数界 + T-I 的 bias 常数 + 部署风险分解。
**验证脚本**：`code/theory/verify_besov_pac.py`（合成 Besov 类上率拟合）。

---

## 3. 逐文件修改清单

### paper/（按执行顺序）
| 文件 | 修改 |
|------|------|
| main.tex | ① Abstract/Intro/Conclusion 重写为理论主线 ② γ/margin/M₂/h 统一符号 ③ 新增符号表节 ④ 所有数字对账（0.079/0.064/0.056/0.13/0.1196 一张表）⑤ "Lemma 1.6"→\ref{lem:per_op}(vi) ⑥ ~50 硬编码 Theorem→\ref ⑦ tab:summary 补 E58 + E21-51 说明 + 去 V7/V1 重复 ⑧ 删除 folklore 叙事 ⑨ Lemma 6 对账（ℓ∞/ℓ₁ 标注）⑩ Thm 1 补 Box-Continuation 假设 ⑪ fixed-point→discretized ⑫ three/four-tier 统一 ⑬ 文献补引（MRW/Tsybakov/CompCert/Nye/Kratsios/NeurIPS-CMI/Giacobbe）⑭ 删除 RBF-KAN/MNIST 实证声称 ⑮ "Tankman" 恢复 \cite 或删除 ⑯ SiLU″ 公式修正（σ(1−σ)[2+x(1−2σ)], sup=0.5）⑰ 实验-修复时间线声明 |
| section_lut_sharp.tex | **重写为 T-I 最优恢复节**（保留 k 阶常数表作为特例） |
| section_svnn_theorems.tex | Thm 6 升级（T-II 下界）+ 新增 T-IV；修 Fixed-Depth 自相矛盾；修 0.0019 复用；Box 延续引理保留 |
| section_trichotomy.tex | necessity 标注全篇一致 + 修正 C² 证据错误 + T-III 结果接入 |
| section_galois.tex | Galois 修复（γ 域加一阶项或重定义）+ 语法碎片修复 + Coq 附录同步 |
| section_svnn.tex | 修正 h=0.75→0.429；deployment margin 定义；Remark 2/3 编号；"cubically"→quadratically；depth-uniform O(L)→O(1) |
| section_characterization.tex | Thm A 宽度指数修正（Θ(εd^{1+L/2})）；fig04 caption d=28 数值修正 + 补引用 |
| section_iec_universal.tex | 通用保证降级为"IEC 60559 操作语义合规控制器"；删"entire ecosystem" |
| section_noninterference.tex | 加 WCET(P)+22.67ms≤cycle 前提；NC.3 溢界修正（[−23.9,25.1]） |
| section_svnn_chebykan.tex | Wavelet-KAN citekey 修正 + 小波类型统一（Mexican-hat）；Fourier-Z3 矛盾调和；Remark 4 编号 |
| references.bib | 补：leroy2009compcert、tsy-bakov2009、micchelli1977survey、gaffney1976optimal、deboor1977computational、nye2025categorical、kratsios2025reskan、giacobbe2020bits + NeurIPS'25 CMI；删 ~24 个未引条目；tankman2026lipschitz 补全 |
| appendix_coq_spec.tex | Galois lemma 修复后重新形式化 |

### code/（按执行顺序）
| 文件 | 修改 |
|------|------|
| neuroplc/differential_test.py | **E67 一行修复**（L183 broadcast：`diff_id.max(axis=1) <= margin_frac*per_in` 或 `per_in[:,None]`）+ 跑通 + 提交 E67 结果 JSON |
| theory/verify_optimal_lut.py | 新建（T-I） |
| theory/verify_compile_aware_minimax.py | 新建（T-II） |
| theory/verify_necessity.py | 新建（T-III） |
| theory/verify_besov_pac.py | 新建（T-IV） |
| theory/（原 4 脚本） | 重跑确认（审稿后无回归） |
| experiments/e19_paderborn.py | 移除/标注 TODO |
| experiments/（新增）e_t1_optimal_grid.py | CWRU 最优网格 vs 均匀网格实证（同精度减半预算） |
| experiments/（新增）e_t2_rate_matching.py | 合成类 n×N 扫描（T-II 率匹配） |
| experiments/（新增）e_t4_besov.py | 合成 Besov 类（T-IV） |
| neuroplc/ir.py | 清理 L445 占位注释 |

### docs/
- CHECKLIST.md（新建）：17 CRITICAL + 全部 MAJOR 追踪表，逐项勾选
- REVIEWER-FAQ.md：按新定理结构 v3 更新
- RESTART_CONTEXT.md：每个里程碑同步
- AUDIT_2026-08-01.md：追加本轮审稿记录

---

## 4. 工业验证规格（双厂商）

| 项 | 工具 | 内容 | 状态 |
|----|------|------|------|
| 西门子编译 | TIA Portal MCP (184) | 重生成全部 SCL（scale 修复后）+ 0e0w | ✅ 已有，Phase 0 复查 |
| 汇川编译 | iFA Evolution MCP | backend_iec.py 输出 ST → iFA 编译 0e0w → plc_sim_download 仿真执行 | 🔴 Phase 0 新实验 |
| 仿真执行 | iFA plc_sim_download | 1000 样本 PyTorch-vs-ST 对比（Tier-4 跨厂商版） | 🔴 Phase 0 |
| 物理 PLC | 可选（S7-1200 实机） | WCET 实测 + 算术偏差 | Phase 3（若借到设备） |

**iFA 实验的论文价值**：从"vendor-neutral（零 token 论证）"升级为"跨厂商实测验证（编译+执行+一致性）"——直接回应 Agent 3 的 CRITICAL（"从未在非西门子工具链编译"）。

---

## 5. 执行时间盒（对齐考研，硬纪律）

| 阶段 | 时间 | 内容 | 退出标准 |
|------|------|------|---------|
| P0 硬修 | 08-03→08-16 | 3 清单（E67/Galois/数字对账/necessity 标注/编号/文献）+ iFA 实验 + MRW 原文阅读 + CHECKLIST.md | 17 CRITICAL 全绿（除需定理的 4 项）；iFA 实验报告 |
| P1 定理 | 08-17→12-31 | T-I(8-9月) → T-II(10-11月) → T-IV(10-12月并行) → T-III(11-12月) | 4 脚本全 PASS + 4 节论文初稿 |
| P2 投稿 | 01-01→03-15 | 论文重构定稿 + review-paper 全量 + 修复 | 0 CRITICAL；2027-03-15 投 TNNLS |
| P3 降载 | 04-01→11-30 | 谱系分类（暑假）+ 可选 Coq；考研全力 | 考研 808 真题 + 数一冲刺 |
| P4 收官 | 2028-01→06 | 复试 + 审稿回复 + VNN-COMP 2028 提案 | 上岸 + 录用 |

---

## 6. 里程碑检查点（每 2 周）

- [ ] 08-16：P0 exit（CHECKLIST 全绿、iFA 报告、MRW 判定）
- [ ] 09-30：T-I 完成（脚本 PASS + 节稿）
- [ ] 11-15：T-II 完成
- [ ] 12-15：T-IV 完成
- [ ] 12-31：T-III 完成或降级决策
- [ ] 02-15：论文重构稿
- [ ] 03-15：投稿 TNNLS

---

## 7. 最终决策点（板板拍板后开工）

1. [ ] 采纳"离散化执行的最优性理论"统一主题（含论文新结构）
2. [ ] 定理优先级 T-I→T-II→T-IV→T-III（T-III 允许降级为"领域公开问题"）
3. [ ] iFA 跨厂商实验纳入 P0（需要 iFA MCP 在线 + 汇川工程环境）
4. [ ] Coq 机械化推迟到 P3（TNNLS 有意向后）
5. [ ] 考研时间盒：P2 投稿后研究降载（2027-04 起）

*最后更新：2026-08-03 | 板板 + Claude | 取代 UPGRADE_PLAN.md Phase 2 部分（其 Phase 1/3/4 并入本文件 3/4/5 节）*
