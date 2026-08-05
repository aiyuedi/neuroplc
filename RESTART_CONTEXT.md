# NeuroPLC — 新窗口上手文档（2026-08-05 v5）

> **若在新 session 中接手本项目，先读此文件，再执行「上手第一步」。**
> v5 由 2026-08-05 更新：证书体系第三轮（soft3L/Fourier sound）+ M21/CMI/conjecture 锚定。
> v4 由 2026-08-05 更新：P2 残余全清（16 overfull hbox → 0）。
> v3 由 2026-08-05 更新：证书体系（双层/有界幅值基）+ Phase 1 审稿闭环（4 P0 全清）。
> 最新快速交接：`HANDOFF_2026-08-05.md`（三分钟版）。

---

## 1. 项目定位（当前）

**论文标题**：A Type Theory of Certifiable Neural Architectures: Sharp LUT Bounds, Complexity Separation, and Compile-Aware Generalization, with an Industrial PLC Instantiation

**目标刊**：IEEE TNNLS（理论为主，PLC为工业案例）

**一句话**：第一个证明"哪些神经网络架构天生可认证、以什么锐常数、什么复杂度、什么验证成本"的容量理论框架——universal packing 线 + 三层验证分层 + 架构即码选择律。

**当前状态**：**91页，0e0w，0 undefined refs，19 定理（Thm 17-19 = 可认证部署容量理论；aux 实证编号 1-19 连续，Thm 12=hot_swap_safety）**，18 commits 双推，**4 P0 审稿阻塞项全清，可送审**

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

### 2026-08-04 深夜~08-05：证书体系 + Phase 1 审稿闭环（commits 561fcc6→278e494）
- **双层证书体系**（section_svnn_theorems E-T9 段）：理论 tier（first-principles 传播修正 dy=Σε + per-edge Lipschitz → **sound 0.288/2.34×**，唯一对未见输入成立的界，δ_fp32≤3×10⁻⁵）+ 实测校准 tier（**validated** 0.058/11.6× f32、0.026/26× f64，公式 0.058=max(0.039, 0.053×1.1)）+ expected 0.00137/493×；main 无证书（2.29/0.3×）
- **有界幅值基证书**（E-T10/11）：FourierKAN 4.9×@99.96%（validated，理论 0.110 不覆盖含超界实测）、WaveletKAN 2.8×@99.67%（sound，理论覆盖实测）——证书几乎免费（0.03/0.26pp vs B-spline 1.4pp）；fig_cert_panorama 三面板
- **Phase 1 审稿闭环**（PRE_SUBMISSION_REVIEW_2026-08-05.md，4 P0 全清）：「sound」措辞纪律（只留 0.288/2.34×）；tab:cross_domain/scalability_grid 重算；da_bounds_summary 口径钉死（expected/worst/validated/sound 四档）；Thm 编号 aux 实证连续；γ²=28.1→237；53.7%→TIA 45.2KB/90.4%；SCL 行数 2,188/2,184；FAQ Q10/Q27-29、cover letter、figure_list v3 同步

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
| `docs/PRE_SUBMISSION_REVIEW_2026-08-05.md` | **Phase 1 审稿报告**（4 P0/14 P1/12 P2 + 修复核对表） |
| `HANDOFF_2026-08-05.md` | 最新快速交接（三分钟上手） |
| `docs/CAPACITY_DESIGN.md` | 容量理论设计文档 v4（三定理 + necessity 链 + 三条红线） |
| `docs/REVIEWER-FAQ.md` | 审稿人防御（Q1-29，含双层证书三层语义） |
| `docs/cover_letter.md` | 投稿信 v2（双层证书 + 有界幅值基） |
| `docs/figure_list.md` | 图清单 v3（17 图） |
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

### 投稿前（按优先级）
1. **TNNLS 页数策略（板板决策，挂起中）**：92 页单栏 ≈ 46 页双栏 vs ~14 页限制——压缩 vs 分流（IMA J. Inf. Inference / FoCM 理论侧；IEEE TII 系统侧），投稿前必须定案
2. ~~**γ<1 全层（soft2L）**~~ ✅ **达成（2026-08-06，verify_bspline_peredge.py）**：per-edge-L 配方 = L0 全局缩放 0.842（保形状，acc 反升 98.70%）+ L1 微调 15ep（每步**实测** γ 投影，E68 口径，eval 模式防 _extend_grid 漂移）→ **γ=[0.95,0.95]（6000 点实测）全层收缩，acc 98.49%（-0.13pp）**；踩坑：解析 row-L 低估 3×（数值差分已含链式 1/3，公式再除 3）、600 点采样投影幻觉（v3 真 γ₂=1.075）、按行投影毁协方差（-7pp vs 全局缩放无损）；诚实边界：解析 row-L=[3.16,1.04] 仍>1（基幅度≥1）→ 证书保持 validated-tier，Fourier/Wavelet sound-tier 优势不变；E-T9 段+tab:cert_thresholds 已同步；v5 checkpoint（kan_contractive_v5.pt）
3. **物理 PLC WCET 实测**（目前 PLCSIM/Z3 模拟 + L0-direct 估算 24.8ms）——诚实限制，cover letter 已声明
4. ~~**P2 残余**~~ ✅ 已全清（c876ce5）；44 underfull 遗留（低优先）
5. **CN 中文版同步**（板板明确"先不理"，5 个 bib key 缺失会编译 [?]）
6. ~~**T1 归约**~~ ✅ **conjecture 升级为定理**（2026-08-06，thm:stratum3-transfer）：单元素嵌入 F={f}∪{0}（ε(S,0)=0）→ 类级可达性 Reach NP-hard（Beliakov 决策形式）+ c_* 计算 NP-hard + ETH 指数下界 + 穷举量化网格上界（2^N log(1/h) 预算达 c_*）+ 结构最优性认证 = 阈值 e*(f,N) ≤ ε(S,f)；与 thm:necessity 互补（实例级耦合 vs 类级自由节点）；7 处 conjecture 措辞同步（stratum-3 段/tradeoff(i)/proof/unification/coordinates/related work）；94 页 0e0w 0 undefined 编译通过

### 已结案（勿再动）
- **证书第三轮 ✅（2026-08-05，904dc16/72c840a/781920d 等）**：
  - **soft3L SOUND 0.284/2.4×**（信号域传播：输出层 per-edge L 4.69→0.29；Box-in 81.4%；修正 0.110 误标 sound 的过度声称）
  - **FourierKAN SOUND 0.130/5.2×**（L0-direct 配置：L0 解析 SIN/COS + L1 LUT N=16 域 6.3；SCL 4,363 行；f32 模拟 maxAE 0.091 被覆盖；WCET 24.8ms/4.0× 余量——e60e61_l0direct.py）
  - **M21 first 断言弱化**（18 处 to our knowledge）、**M6 注释编号修复**、**CMI 引用**（Sefidgaran NeurIPS 2025 Oral, arXiv:2510.23485 + Thm 6 关系段）
  - **c_*/T1 conjecture 锚定**（free-knot 最优性 NP-hard 文献：Mohr 2023 set-partitioning + Beliakov 2004）
  - **cross_domain 表全填**（Cond.3：MNIST/ChebyKAN 实测 No；M2 max：MNIST 0.83/XJTU 27.1）
- **16 overfull hbox 全清 ✅（2026-08-05，c876ce5）**：tab:xjtu_ft 189pt（5 列声明+脚注移出）、tab:cert_thresholds 79pt（7 列声明+Evidence p 列）、长代码路径段落 4 处（局部 tolerance 组）、Thm 5/6 段落、FourierKAN/WaveletKAN 列宽、Step4/TableIX 微溢出
- FIXME/TODO 清扫 ✅（2026-08-04 完成 18 处）
- tab:summary E-T 行 ✅（E-T1-E-T11）
- Box-Continuation ✅（soft 模型 99.9% 覆盖，main 0%——收缩训练才是钥匙，加宽域收益有限）
- 定理重编号 ✅（aux 实证 1-19 连续无跳号，容量=17-19，不再动）
- 有界幅值基全收缩 ✅（Fourier/Wavelet 第一层 γ<1，第二层仍>1——诚实限制已入 cover letter）
- 三分法 necessity：CP 内定理 + 绝对形式开放问题定位 ✅（保持）

---

## 5. 三个诚实底线（勿动）

1. **训练网络非收缩（γ=[15.4,5.3]）**——E-T9 软收缩演示（γ→[1.03,1.38]，11× 改善）已诚实写入；全收缩受 B-spline 基幅值限制（架构常数，未来工作）
2. **三分法 necessity 全篇 conjecture 标注**——CP 内相对形式是定理；绝对形式（P≠NP 级）是开放问题——这是贡献不是缺陷
3. **手写编号脆弱**——插新定理必须手动 bump 下游计数器（setcounter 链，本次踩坑 2 次）

---

## 6. 上手第一步（新session立刻执行）

```
1. 读 HANDOFF_2026-08-05.md（三分钟版）与本文件
2. cd D:/neuroplc-paper && git log --oneline -5   # 确认 HEAD（18 commits 双推，最新 278e494）
3. 编译验证：cd paper && xelatex -interaction=nonstopmode -halt-on-error main.tex（两遍）
   → 期望：91页，0 errors，0 undefined refs
4. 读 docs/PRE_SUBMISSION_REVIEW_2026-08-05.md（审稿状态）与 HANDOFF_2026-08-05.md（证书矩阵）
5. 按「剩余待办」继续：页数策略（板板）→ T1 归约（独立 session）→ 投稿
```

---

## 7. 环境备忘

- Python 3.14.3（系统）+ `D:\dev-tools\research\venv\`（含 torch/matplotlib/scipy/sklearn）；模型路由 V4-Flash
- 论文编译：MiKTeX xelatex（`D:\miktex\`）
- 模型：`results/student/kan_kd_vrmKD_best.pt`（主，99.93%）；`kan_contractive.pt`（E-T9 软收缩，98.5%）
- 数据：`data/processed/features_X.npy`（CWRU 28-D，13714 样本）
- Git：origin=Gitee, github=GitHub（双推工作流）

*最后更新：2026-08-05 | 板板 + Claude | 91页 0e0w | 19 定理 | 18 commits 双推 | 容量理论 + 证书体系 + Phase 1 审稿闭环（可送审）*
