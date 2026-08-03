# NeuroPLC — 新窗口上手文档（2026-08-03）

> **若在新 session 中接手本项目，先读此文件，再执行「上手第一步」。**
> 上一个工作窗口完成了从"修复错误"到"领域奠基级升级"的全过程（12 commits 已推 Gitee+GitHub）。

---

## 1. 项目定位（当前）

**论文标题**：A Type Theory of Certifiable Neural Architectures: Sharp LUT Bounds, Complexity Separation, and Compile-Aware Generalization, with an Industrial PLC Instantiation

**目标刊**：IEEE TNNLS（理论为主，PLC为工业案例）

**一句话**：第一个证明"哪些神经网络架构天生可认证、以什么锐常数、什么复杂度、什么泛化保证"的类型理论框架。

**当前状态**：**82页，0 errors, 0 warnings, 0 undefined refs，TNNLS定位重组完成**

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

## 3. 关键文件索引

| 文件 | 用途 |
|------|------|
| `paper/main.tex` | 主文件（82页，0e0w） |
| `paper/section_svnn.tex` | SVNN理论（Thm 2/8 + Corollary 3 DA最优性） |
| `paper/section_lut_sharp.tex` | **Thm 9p**（锐常数） |
| `paper/section_trichotomy.tex` | **Thm 10p**（三分法） |
| `paper/section_svnn_theorems.tex` | **Thm 5p/6p** + Box延续引理 + 分辨率匹配 |
| `paper/section_svnn_chebykan.tex` | ChebyKAN（Prop 3） |
| `paper/references.bib` | 91条（已清理） |
| `code/neuroplc/frontend.py` | scale折叠修复点（L234-264） |
| `code/neuroplc/differential_test.py` | Tier 4差分测试器 |
| `code/neuroplc/backend_iec.py` | 多厂商IEC后端 |
| `code/theory/verify_*.py` | 4个定理验证脚本（全PASS） |
| `results/theory/*.json` | 验证结果（thm9p/lemma3p/thm5p/thm10p/thm6p） |
| `docs/AUDIT_2026-08-01.md` | 原始审查记录+修复后状态 |
| `docs/REVIEWER-FAQ.md` | 审稿人防御文档（需更新新定理FAQ） |

**验证脚本速查**：
```bash
cd code
python theory/verify_sharp_constants.py      # Thm 9p
python theory/verify_lemma3_bennett.py       # Lemma 3p
python theory/verify_thm5_eps_separation.py  # Thm 5p
python theory/verify_trichotomy_thm10p.py    # Thm 10p
python experiments/e59_thm6p_compile_aware.py  # Thm 6p（约7分钟）
python -m neuroplc.differential_test         # Tier 4（需模型）
python generate.py                           # 重新生成SCL
cd paper && xelatex main && bibtex main && xelatex main && xelatex main  # 编译
```

---

## 4. 剩余待办（下一步优先）

### 投稿前（高优先）
1. **REVIEWER-FAQ更新**：现有FAQ是旧版（12问），需补充新定理的防御（锐常数、三分法、ε-分离、编译感知PAC、Tier 4、算法4、多后端）
2. **`review-paper`深度审稿**：最后一道保险（6-Agent），验证0 CRITICAL
3. **E63-E68实验脚本归档确认**：确保每个实验编号有对应脚本+JSON（E63=verify_sharp_constants, E64=verify_lemma3_bennett, E65=verify_thm5_eps_separation, E66=verify_trichotomy_thm10p, E67=differential_test, E68=e59_thm6p_compile_aware）
4. **投稿材料**：cover letter、author bio、图清单（TNNLS要求）

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
2. cd D:/neuroplc-paper && git log --oneline -15   # 确认在 6de2d0b
3. 编译验证：cd paper && xelatex main && bibtex main && xelatex main && xelatex main
   → 期望：82页，0 errors，0 undefined refs
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

*最后更新：2026-08-03 | 板板 + Claude | 82页 0e0w 投稿就绪状态*
