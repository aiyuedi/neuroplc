# NeuroPLC 升级计划：开创级 → 奠定级 → 王者级（2026-08-03）

> 依据：7-Agent 深度审稿（PRE_SUBMISSION_REVIEW_2026-08-03.md）+ 两轮联网情报调研（2026-08-03）。
> 当前评级：**Incremental**（Agent 6, TNNLS）。目标：开创级（field-founding）。

---

## 0. 根因诊断（5-Why）

1. 表面：4 个锚定定理被指为"教科书重述 / 经典律 / 猜想"
2. 直接原因：锐常数 = 经典插值余项定理；分辨率匹配 = 非参数估计经典带宽律（Tsybakov）；necessity = 猜想
3. 系统原因：理论构建是**已有结果的组合**而非**新问题的求解**——问题是别人提过的，只是换了应用场景
4. 流程原因：理论是从编译器项目（系统贡献）"拉出来贴标签"的，不是从理论问题出发
5. **根本原因：缺少一个"别人没提过、回答了会改变领域看法"的理论问题**

**开创级判据**（自检清单）：
- (a) 提出了一个新问题（不是新答案给旧问题）
- (b) 证明了该问题的核心定理（不是 conjecture）
- (c) 后续工作必须引用它才能进入该方向（成为起点）

---

## 1. 情报调研结论（2026-08-03，两轮 11 次搜索）

### 机会确认（文献空缺）
| 方向 | 现状 | 对 NeuroPLC 的意义 |
|------|------|---------------------|
| KAN 验证 | VNN-COMP 2025（8队/25基准）全 ReLU/ONNX，**KAN 零覆盖**；"Verifying Robust KANs"(IEEE 2025) 只做对抗鲁棒 | **编译验证方向空白**——差异化主场 |
| 最优恢复理论 | Micchelli-Rivlin-Winograd (1974)：spline 插值最优、最优结序列=Chebyshev perfect spline 零点、锐常数=其范数；de Boor optknt 数值构造 | **均匀 LUT 网格 minimax 次优**——Thm 9(3) 从断言变定理的理论武器库 |
| 量化/离散化验证 | QEBVerif(DRA+MILP)、QA-IBP、Frame Quantization(ΣΔ,JFAA'25)、DiscQuant(COLT'25) | 全是**权重/激活定点量化**；**B-spline 函数离散化（LUT）验证空白** |
| 类型理论框架 | Nye 2025 (arXiv:2508.11647) Lawvere+Para 逻辑架构；Isabelle/AFP 形式化 NN；rocq-transformer；Ionia | 都是"逻辑/维度正确性"；**"数值可认证性类型理论"可区隔，但必须引用 Nye 并明示差异** |
| NN→PLC 已验证编译器 | Werner 2025 博士论文(PLCreX)验证 PLC 代码本身；Boudribila 2025 AI 生成 PLC 验证=future work | **"NN→IEC 61131-3 已验证编译器"空缺确认**（TII/TC 路线强卖点） |
| 部署感知泛化 | NeurIPS 2025 ε-lossy compression + CMI 界（唯一近邻） | 编译感知 PAC 依然空，但须引用并定位差异 |
| KAN 逼近理论 | Kratsios-Kim-Furuya (Neural Networks 2025)：Res-KAN Besov 最优率 + 伪维数无诅咒样本复杂度 | **编译 bias 应与 Besov 正则结合**——新定理来源 |
| folklore 1/8 | 全网无出处 | 支持审稿判定：**删除"修正 folklore"叙事** |

### 关键风险（需在 Phase 2 前用原文献验证）
- R1：最优网格定理是否已被 Micchelli-Rivlin 覆盖？→ 差异化关键：**LUT 表格约束（离散存储、逐节点字长）下的恢复问题是否产生新常数**。必须读 MRW 1974 原文（opencite 可取）。
- R2：Nye 2025 与"类型理论"命名冲突 → 论文须显式区隔"逻辑正确性 vs 数值可认证性"。
- R3：necessity 与 P≠NP 纠缠 → 约束类（affine 证书类）下的 necessity 是可行路径，全必要性保留为公开问题（本身可成为领域开放问题）。

---

## 2. 三级目标定义

### 开创级（field-founding）— 论文成为"NN 编译验证理论"的起点
- 提出新问题："**哪个架构类在编译到离散执行（LUT）意义下可认证，且以什么最优常数、什么复杂度、什么泛化最优权衡？**"
- 核心定理（全部证明 + 验证脚本 + 极值族）：
  - **定理 I（最优恢复版锐常数）**：LUT 约束下最优恢复常数 = perfect-spline 范数；**最优节点放置定理**（非均匀最优网格 + 与均匀网格的显式 gap 常数）+ 工程可用算法（de Boor optknt 的 LUT 版）
  - **定理 II（编译感知 minimax 最优权衡）**：N*(n) 从 heuristic 升级为渐近最优性（含下界：不存在更少 LUT 达到同部署风险的编译方案）
  - **定理 III（necessity 第一格）**：在"design-time-computable affine 证书"类下证明三分法 necessity（非平凡子类）
  - **定理 IV（Besov 编译感知 PAC）**：离散化 bias × Besov 正则 → 维度无关样本复杂度（对接 Kratsios 2025）

### 奠定级（foundational）— 领域标准参考
- 复杂度谱系全图：常见激活类（ReLU/SiLU/GELU/多项式/三角/B-spline/小波）验证复杂度分类定理
- Coq 机械化：Galois 修复后重新形式化 + 锐常数/最优网格定理机械化
- 收缩训练端到端演示（谱归一化 γ<1）+ 物理 PLC 实测（S7-1200，WCET 四值对账）
- 跨厂商编译验证（CODESYS/TwinCAT 至少一个）

### 王者级（benchmark/SOTA dominance）— 领域"同义词"
- 发起 **LUT 编译验证基准**：格式对接 VNN-LIB 2.0（其 "network theory" 抽象正支持新网络类）→ 成为 VNN-COMP 2027 新赛道 → 领域命名权
- 工具开源 + 与 QEBVerif/Frame Quantization 等全量对比竞速
- IEC 61508 / ISO/IEC TS 22440 标准相关产出
- 论文组合拳：TNNLS 理论篇 + TII 系统篇 + 基准篇（3-4 篇）

---

## 3. 分阶段执行计划

### Phase 1：硬伤清零 + 叙事重定位（1-2 周，不修必死）
| # | 任务 | 出处 |
|---|------|------|
| 1 | E67 一行修复（differential_test.py:183 broadcast）+ 跑通 + 提交结果 JSON | Agent 7 FAIL |
| 2 | Theorem B Galois 修复（加一阶项或重定义 γ 域）+ Coq 附录同步 | Agent 3/4 CRITICAL |
| 3 | Lemma 6/Thm 1 对账：标注 ℓ∞ vs ℓ₁，统一三值（MaxAE 3.65/14.17/≈17） | Agent 2/4 CRITICAL |
| 4 | γ 定义区分（收缩 vs 放大 vs 安全因子），全篇一致化 | Agent 2/3/4 CRITICAL |
| 5 | necessity 标注全篇一致化（摘要/引言/结论/标题 + 修正 C² 证据错误） | Agent 3 CRITICAL |
| 6 | 数值对账表重做（DA 0.079/0.064/0.056/0.13/0.1196 + M₂ 校准说明） | Agent 2/4 |
| 7 | 文献补引：CompCert/Leroy、Tsybakov、Micchelli-Rivlin、Giacobbe、**Nye 2025**、**Kratsios 2025**、NeurIPS'25 CMI；Tankman 恢复或删除 | Agent 1/2/6 |
| 8 | 删除"修正 folklore"叙事 → "显式窗口积常数 + 认证极值族" | Agent 6 CRITICAL |
| 9 | 编号/术语/表格清理：Lemma 1.6→\ref、~50 硬编码→\ref、Remark 碰撞、SiLU″ 公式、fixed-point 措辞、three/four-tier、tab:summary E58/E21-51/V7 重复 | Agent 1/2/4/5 |
| 10 | **路线决策**：TNNLS（理论深修）vs TII/TC（系统故事）→ 决定 Phase 2 侧重 | 板板决策 |
| 11 | 双推所有 commits | Agent 7 WARN |

### Phase 2：开创级核心定理（1-3 月，真正的研究）
| # | 任务 | 方法 | 验证 |
|---|------|------|------|
| 1 | 读 MRW 1974 / Gaffney-Powell / de Boor optknt 原文（opencite）→ 确认 R1 | 文献 | 差异点文档 |
| 2 | 定理 I：LUT 最优恢复 + 最优节点放置 + 均匀 gap 常数 | perfect-spline 对偶 + 数值极值 | verify_optimal_lut.py + JSON |
| 3 | 定理 II：编译感知 minimax 下界 | 统计 minimax + 插值下界对偶 | verify_compile_aware_minimax.py |
| 4 | 定理 III：necessity 第一格（affine 证书类） | 复杂度归约 + 锐常数下界 | verify_necessity.py |
| 5 | 定理 IV：Besov 编译感知 PAC | Kratsios 率 + 编译 bias 组合 | verify_besov_pac.py |
| 6 | 每定理：验证脚本 + 极值族 + JSON（复用现有 results/theory 基础设施） | — | 全 PASS |

### Phase 3：奠定级（3-6 月）
- 复杂度谱系分类定理（全部常见激活类）
- Coq 机械化（Galois 修复版 + 定理 I 最优网格）
- 谱归一化收缩训练端到端（γ<1, ≥99%）+ 全链验证
- 物理 S7-1200 实测（WCET 对账 + 算术偏差）
- CODESYS/TwinCAT 至少一次编译验证

### Phase 4：王者级（6-12 月）
- VNN-LIB 2.0 "network theory" 对接 → LUT 编译验证基准赛道提案
- 工具开源 + 生态文档 + 与量化验证 SOTA 全量对比
- 标准工作（IEC 61508 / TS 22440 相关）
- 论文矩阵：TNNLS 理论 / TII 系统 / 基准论文

---

## 4. 工具链映射（科研工具路由）
- 文献：lit-review（Express 模式）+ opencite（DOI/PDF/卡片）+ dp-grounding（声称核验）
- 理论验证：code/theory/ 新脚本（沿用 verify_*.py 模式）
- 审稿：review-paper（每阶段后全量跑）
- 写作：lightchuan-paper（中文）/ academic-writing-skills（英文 IMRAD）
- 图表：lightchuan-figure-studio / Origin MCP
- 记忆：engram + memory 文件（本计划已入 D:\neuroplc-paper\UPGRADE_PLAN.md）

---

## 5. 决策点清单（需要板板拍板）
1. **路线**：TNNLS 理论深修 vs TII/TC 系统主线（Phase 1 #10）——建议：双线并行准备，Phase 2 定理做出来再定投
2. **时间盒**：Phase 2 的 4 个定理，建议先做定理 I（最优网格——最快见效、最差异化），定理 III 最后（最难）
3. **资源**：Coq 机械化（Phase 3）是否需要，取决于目标刊（TNNLS 加分显著，TII 不需要）
4. **基准赛道**（Phase 4）：是否投入取决于 Phase 2 成果质量

*最后更新：2026-08-03 | 板板 + Claude | 基于 7-Agent 审稿 + 11 次联网检索*
