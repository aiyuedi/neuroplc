# NeuroPLC — 新窗口上手文档（2026-08-03）

> **若在新 session 中接手本项目，先读此文件，再执行「上手第一步」。**
> 上一个工作窗口完成了从"修复错误"到"领域奠基级升级"的全过程（12 commits 已推 Gitee+GitHub）。

---

## 1. 项目定位（当前）

**论文标题**：A Type Theory of Certifiable Neural Architectures: Sharp LUT Bounds, Complexity Separation, and Compile-Aware Generalization, with an Industrial PLC Instantiation

**目标刊**：IEEE TNNLS（理论为主，PLC为工业案例）

**一句话**：第一个证明"哪些神经网络架构天生可认证、以什么锐常数、什么复杂度、什么泛化保证"的类型理论框架。

**当前状态**：**87页，0e0w，0 undefined refs，19 定理（Thm 17-19 = 可认证部署容量理论，2026-08-04 P3 重组完成）**

---

## 2. 完成的工作（12 commits，已推送）

### 层1：修复（commit 11e81e6）
- **scale bug**（最严重）：`backend_s7.py`的`_emit_add`不乘scale_base/scale_spline（训练后≈1.2-1.7），真实SCL分类一致性仅83%。修复：`frontend.py`折叠scale进权重（`base_weight*scale_base`、`table*scale_spline`）→ 重新生成全部SCL → 验证100%一致，MaxAE 0.52
- **7条伪造引用**删除（含`szasz2025soundness`→真实版`szasz2025floating`）；markov年份修正；lutkan2025→2024
- **E53根因**：层间特征越界（h0超出[-3,3]）→ LUT钳位vs精确B样条分歧，16.8恒定之谜解开——支撑Box延续引理

### 层2：升级定理（commit 8b36093）——"修复即创新"
| 定理 | 升级内容 | 验证 |
|------|---------|------|
| **Thm 9p** (section_lut_sharp.tex) | 锐k阶LUT常数：c_k=Q_k/k!（c₃=1/(9√3)≠folklore 1/8；c₄=1/24），极值族精确达到界 | verify_sharp_constants.py → ratio=1.000000 |
| **Lemma 3p** (main.tex) | Bennett矩鲁棒DA收紧（无均匀幅值假设，只需κ=ν/μ=O(1)），necessity反例族 | verify_lemma3_bennett.py → medR随√d 4.6→37.7 |
| **Thm 5p** (section_svnn_theorems.tex) | ε-分离：产品门3-SAT自包含归约（含分数一致性），SVNN poly vs 耦合NP-hard，Pareto推论 | verify_thm5_eps_separation.py → 200轮+时序5719x |
| **Thm 6p** (section_svnn_theorems.tex) | 编译感知PAC：R(N̄)=R̂(N)+gap(γ^{L-1}/√n)+bias(γ^{L-1}c_kM_kh^k)，分辨率匹配定律N*≍n^{1/(2k)} | e59_thm6p_compile_aware.py → 分解确认，锐常数0.996-1.000 |
| **Thm 10p** (section_trichotomy.tex) | 可验证性三分法：SVNN是(P1)(P2)(P3)唯一共同满足域；necessity为conjecture+强证据 | verify_trichotomy_thm10p.py → 斜率-2 vs -1分离 |
| **Box延续引理** | 域包含问题的poly解法（E53/E68越界实证支撑） | 已写入 |

### 层3：创新（commit 64d6595, f6f5750, a7a0d78）
- **Tier 4 编译器自验证**：`code/neuroplc/differential_test.py`差分测试（PyTorch vs SCL语义模拟），`compiler.py compile(verify=True)`集成（失败拒部署）——**抓住了真实scale bug**
- **算法4**（main.tex）：运行时LUT热切换（双缓冲+影子验证+回滚窗口），降级保证定理
- **多后端**：`code/neuroplc/backend_iec.py` vendor-neutral IEC 61131-3 ST（0个Siemens token，语义验证PASS）
- **E68实证**：编译感知PAC验证（分解+锐常数确认+诚实发现）

### 层4：TNNLS重组（commit ebc472d, d268393, 23dc2cc, 27045af, 6de2d0b）
- 新标题/Abstract/Intro/贡献/结论（理论为主）
- 两轮深度审稿修复：14条一致性（Theorem 10/Lemma 3编号冲突、旧表述残留）+ 14条数学（Q₃=2/(3√3)、极值族C^k、N*指数、三分法conjecture）
- E63-E68实验表格行补充

---

## 2.5 2026-08-03 P0 硬修（7-Agent 深度审稿 → 10 commits，已双推）

**背景**：新窗口按本文件接手后，先跑 7-Agent 深度审稿（`PRE_SUBMISSION_REVIEW_2026-08-03.md`），再按 `FINAL_PLAN.md` 执行 P0。

**核心发现链**（审稿没挖到的深层问题）：
1. **E67 不可复现** → 三层修复：broadcast bug + 真实测试特征 + **模拟器语义 bug**（np.interp 钳位 vs SCL 外推）→ 3357 样本 100% 一致 maxAE 0.47
2. **γ=0.182 是陈旧手写断言**（无代码支撑，因子全无法重现）→ 实测 L_B=2.21/‖W_eff‖=12.36/6.52，γ=[15.4,5.3] 正确
3. **界值全部重算**（`code/theory/verify_da_bounds_recomputed.py`）：Δ_IA 0.172→**1.38**（证书失效）、Δ_DA 0.079→**0.66**（8.5×→**1.0×** 边缘）、M₂^max 版 5.9 → "契约性训练是部署必需"有了数据支撑
4. **WCET 22.67ms 用旧时序** → 重跑 3.86ms（三层聚类 2.86/3.2/3.86，故事更强）

**其他**：Galois 连接修复（α 加一阶项+Coq 同步）、Lemma 1.6→\ref、SiLU″ 修正（sup=0.5）、0.182 全清除、43 处数字同步（MLP 38×→4.6×）、necessity 全篇 conjecture 标注、WCET 时序统一、MNIST 列补入（E42 98.59%）、Tankman 删除、fixed-point 措辞等。

**待重生成图**（FIXME 已标）：fig02/fig04/fig05/fig09 + fig1_overview（数值已变，图内数据旧）。

## 2.6 2026-08-03 P1 四定理完成（最强模式，同日）

| 定理 | 位置 | 内容 | 验证 |
|------|------|------|------|
| **T-I** Thm 13 | section_lut_sharp.tex §sec:optimal-lut | 曲率感知两级最优 LUT：跨函数 n_i∝M₂^{1/2}（2.75×）+ 函数内 ρ∝√M₂(x)（3.32×） | verify_optimal_lut.py PASS；**发现并修复 optimizer.py κ 密度 bug（劣于均匀 2.1×）** |
| **T-II** Thm 14 | section_svnn_theorems.tex §sec:compile-aware-minimax | R(n,N)≍max{n^{-2k/(2k+1)}, e*(N)}：样本-预算剪刀差 + 离散编译免费 | verify_compile_aware_minimax.py PASS（1.003 平坦 / 269× 衰减） |
| **T-III** Thm 16 | section_trichotomy.tex §sec:necessity-first | necessity 第一格（affine 证书类）：kink 无证书 + 乘积耦合 NP-hard + 锐率⟹最优分配 | verify_necessity_first.py PASS；**修正 trichotomy kink 率论证真实错误**（单函数任意小 vs 类宽度 slope -1.05） |
| **T-IV** Thm 15 | section_svnn_theorems.tex §sec:besov-pac | Besov 光滑度自适应率：bias 超收敛 N^{-2min(s,k)}、预算指数随 s 减小 | verify_besov_pac.py PASS（N=32 衰减 1.4×/6.5×/9.1× 随 s） |

**复跑审稿（2026-08-03 晚）**：✅ 完成——抓出 7 个真实缺陷全修（Thm13 闭合解/Thm14-15 指数/Thm16(iii)/编号 off-by-one/WCET 推导/E52 重算）+ 7 Minor（SiLU 公式/E57 8.3×/ETH 标注等）；84 页 0e0w。
**图重生成**：✅ 完成（12 .m 更新 + 56 图文件，MATLAB 验证）。
**剩余**：投稿材料（cover letter/bio/图清单）、REVIEWER-FAQ v3（新定理）、C11（加宽 LUT 实验，可 P3）。

## 3. 关键文件索引

| 文件 | 用途 |
|------|------|
| `paper/main.tex` | 主文件（84页，0e0w，Thm 1-16） |
| `paper/section_svnn.tex` | SVNN理论（Thm 2/8 + Corollary 3 DA最优性） |
| `paper/section_lut_sharp.tex` | **Thm 9p**（锐常数）+ **Thm 13**（T-I 最优 LUT） |
| `paper/section_trichotomy.tex` | **Thm 10p**（三分法）+ **Thm 16**（T-III necessity 第一格） |
| `paper/section_svnn_theorems.tex` | **Thm 5p/6p** + **Thm 14**（T-II minimax 率）+ **Thm 15**（T-IV Besov 率）+ Box延续 |
| `paper/section_capacity.tex` | **容量理论（P3 新增，2026-08-04）**：Thm 17 分层（universal packing 线 + 三层 + CP 内 necessity）、Thm 18 权衡（常数购买/类适配免费/HLZ 例外）、Thm 19 架构选择律 + CP 证明系统 + related work 边界 |
| `paper/section_svnn_chebykan.tex` | ChebyKAN（Prop 3） |
| `paper/references.bib` | 91条（已清理） |
| `code/neuroplc/frontend.py` | scale折叠修复点（L234-264） |
| `code/neuroplc/differential_test.py` | Tier 4差分测试器 |
| `code/neuroplc/backend_iec.py` | 多厂商IEC后端 |
| `code/theory/verify_*.py` | 8个验证脚本（全PASS，含 4 新定理） |
| `results/theory/*.json` | 验证结果（thm9p/lemma3p/thm5p/thm10p/thm6p） |
| `docs/AUDIT_2026-08-01.md` | 原始审查记录+修复后状态 |
| `docs/REVIEWER-FAQ.md` | 审稿人防御文档 **v3（20问，含新定理）** |
| `docs/cover_letter.md` | 投稿 cover letter 草稿 |
| `docs/CHECKLIST.md` | P0 追踪表（17 CRITICAL 全绿） |

**验证脚本速查**：
```bash
cd code
python theory/verify_sharp_constants.py      # Thm 9p
python theory/verify_lemma3_bennett.py       # Lemma 3p
python theory/verify_thm5_eps_separation.py  # Thm 5p
python theory/verify_trichotomy_thm10p.py    # Thm 10p
python experiments/e59_thm6p_compile_aware.py  # Thm 6p（约7分钟）
python theory/verify_optimal_lut.py             # T-I（Thm 13）
python theory/verify_compile_aware_minimax.py   # T-II（Thm 14）
python theory/verify_besov_pac.py               # T-IV（Thm 15）
python theory/verify_necessity_first.py         # T-III（Thm 16）
python theory/verify_da_bounds_recomputed.py    # 界值重算
python theory/verify_capacity.py                # 容量 packing 线常数匹配（KT 极值 1.000000 / 排序 c_*<c_fix<c_k）
python theory/verify_stratification.py          # 三层分离（kink -1.11 vs C2 -2.05 / 验证成本解耦 vs 超线性）
python theory/verify_decision_law.py            # 架构选择律相变（exp_slope=-ρ 精确匹配 / W1 vs 解析类）
python -m neuroplc.differential_test         # Tier 4（需模型）
python generate.py                           # 重新生成SCL
cd paper && xelatex main && bibtex main && xelatex main && xelatex main  # 编译
```

---

## 4. 剩余待办（下一步优先）

### 投稿前（高优先）
0. **P3 容量理论重组（2026-08-04 完成）**：Thm 17-19 + CP 证明系统 + 三验证脚本全 PASS；**87 页 0e0w**；设计文档 `docs/CAPACITY_DESIGN.md` v4（含三条红线）；容量章节 related work 以 Gappa/Sollya/LeanCert/PCC 为起点（红线①）
1. **REVIEWER-FAQ更新**：🔴 **v4 待**（容量理论防御：Thm 17-19 + CP 定位 + 验证侧/逼近侧领域划分 + Gappa/SNARK 撞名防御）
2. **`review-paper`深度审稿**：✅ 已跑两轮（7-Agent 初审 + 复跑验证修复）；**P3 重组后需最终全量一轮**
3. **E63-E68实验脚本归档确认**：✅ E67 已修复+JSON 落盘；E63-E66/E68 全 PASS
4. **P0 剩余**：C11（加宽 LUT 实验）、C12（编译器边界声明）、M2 剩余（MRW 1976/CompCert/Nye 2025——容量新引用已补 15 条）、M26（iFA 跨厂商实验）、图重生成（fig02/04/05/09/fig1_overview）
5. **投稿材料**：cover letter ✅ 草稿（容量理论加入后需更新）；author bio、图清单 待

### 理论深化（中优先）
5. **Thm 10p necessity 从 conjecture 升级为正式证明**（三分法的necessity方向——现在是强证据）
6. **物理PLC测量**（论文已列为未来工作；PLCSIM等价性论证可先写严谨化）
7. **B-spline knot-aligned LUT的精确复现**——可作为工程卖点（ε=0）

### 已知限制（诚实保留）
- 中文版 main_cn.tex 冻结于7-10（已决策暂缓，**如重建需按英文版结构重写定理环境**）
- 训练网络不收缩（γ=[15.4,5.3]）——论文已诚实处理
- 层2 LUT域越界（47%激活超[-3,3]）——Box延续引理+表加宽是部署要求
- 无物理PLC测量（PLCSIM+Z3为主）

---

## 5. 上手第一步（新session立刻执行）

```
1. 读本文件（你正在读）
2. cd D:/neuroplc-paper && git log --oneline -15   # 确认 HEAD（今日 27 commits，最后 bd56791）
3. 编译验证：cd paper && xelatex main && bibtex main && xelatex main && xelatex main
   → 期望：84页，0 errors，0 undefined refs
4. 读 docs/AUDIT_2026-08-01.md 的状态更新段（修复后状态）
5. 按「剩余待办」继续：先更新 REVIEWER-FAQ → 跑 review-paper 深度审稿
```

---

## 6. 环境备忘

- Python 3.14.3（系统）+ `D:\dev-tools\research\venv\`；模型路由 V4-Flash
- 论文编译：MiKTeX xelatex（`D:\miktex\`），命令见上
- 模型：`results/student/kan_kd_vrmKD_best.pt`（KAN [28,16,4]）
- 数据：`data/processed/features_X.npy`（CWRU 28-D特征，13714样本）
- Git：origin=Gitee, github=GitHub（双推工作流，见 [[gitee-github-push-workflow]]）

*最后更新：2026-08-04 | 板板 + Claude | 87页 0e0w | 19 定理（P3 容量理论）| 投稿材料待更新（FAQ v4/cover letter）*
