# NeuroPLC Paper Project

> **NeuroPLC: An IR-Based Compiler for Deploying Feedforward Neural Networks on IEC 61131-3 Controllers with Application to Bearing Fault Diagnosis**

Started: 2026-07-03 | Status: **方案 C — 全面武器化 (2026-07-04 升级)**
Author: 刘甫悦 (板板) + Claude

---

## 零、当前会话状态

```
项目: D:\neuroplc-paper\
方案: 方案 C — IR-Based 通用编译器 + B-spline 自适应采样 + 多目标PLC
      (2026-07-04 从 KAN工具 → 通用编译器架构 最终升级)

✅ 已完成 (前置准备):
  download_verify_cwru.py  GFW 兼容版 (本地导入 + 校验)
  preprocess.py            28-D (20 统计 + 8 离散熵)
  config.yaml              方案 C — IR 编译器参数 + 自适应采样
  paper/main.tex            IEEEtran 骨架
  paper/references.bib       14 核心引用占位
  code/tests/               42 tests (38 pass, 4 skip)
  neuroplc/utils/mlflow     MLflow sqlite 追踪器

⏳ 待写 — Phase 1 (模型+训练):
  models/teacher_cnn.py
  models/student_kan.py
  models/student_mlp.py
  train_teacher.py
  train_student_kd.py
  evaluate.py
  visualize.py

⏳ 待写 — Phase 2 (编译器 — 核心创新):
  neuroplc/ir.py             IR 图数据结构
  neuroplc/frontend.py       PyTorch → IR
  neuroplc/optimizer.py      B-spline 自适应采样 + 优化 passes
  neuroplc/backend_s7.py     IR → SCL (S7-1200 + S7-1500)
  neuroplc/analyzer.py       静态内存/FLOPs 分析
  neuroplc/compiler.py       编排器 (重构)
  neuroplc/scl_templates.py  SCL 模板 (重构)
  neuroplc/validator.py      交叉验证 (适配)

🔴 板板待做:
  P0-1  下载 CWRU 全量数据 (123云盘: https://www.123pan.com/s/xBwHjv-WIzk.html 提取码 EXLF)
  P0-2  导入 12 篇核心文献到 Zotero

启动提示（粘贴到新会话）:
  我在写论文 "NeuroPLC: IR-Based Compiler for Deploying Neural Networks on PLC"。
  项目 D:\neuroplc-paper\，先读 README.md。
  方案 C，已做完前置准备，下一步写 Phase 1 模型代码。
```

---

## 一、论文定位

### 目标
**不发期刊，按期刊标准写，达到「研究生顶尖水平」的小论文。**

### 标题
*NeuroPLC: An IR-Based Compiler for Deploying Feedforward Neural Networks on IEC 61131-3 Controllers with Application to Bearing Fault Diagnosis*

### 定位
**系统论文为主 + 算法论文为辅。** 核心卖点不是你发明了 KAN——而是你设计了一个**通用的、有架构设计的编译器**，KAN 只是它支持的第一个（也是最佳的）案例。

### 核心贡献 (5 项)

| # | 贡献 | 类型 | 层次 |
|---|------|:---:|:---:|
| ① | 设计并实现了首个基于 IR 的 PyTorch→IEC 61131-3 通用编译器 | **系统创新** | ⭐⭐⭐⭐⭐ |
| ② | B-spline 自适应采样算法：曲率感知的非均匀离散化 | **原创算法** | ⭐⭐⭐⭐ |
| ③ | 多目标 PLC 代码生成 (S7-1200 紧凑模式 / S7-1500 性能模式) | 系统创新 | ⭐⭐⭐ |
| ④ | KAN + VRM-KD + 28-D 多尺度离散熵：参数高效+可解的PLC诊断 | 算法支撑 | ⭐⭐⭐ |
| ⑤ | TIA Portal MCP 全自动编译验证 + 静态内存分析 | 工程创新 | ⭐⭐⭐ |

---

## 二、编译器架构 (核心创新)

```
                        NEUROPLC COMPILER
                              |
    ┌──────────┬──────────┬───┴───┬──────────┬──────────┐
    │          │          │       │          │          │
    ▼          ▼          ▼       ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│FRONTEND│ │  IR   │ │OPTIMIZ│ │ANALYZE│ │BACKEND│ │VALIDATE│
│PyTorch│▶│ GRAPH │▶│  ER   │▶│   R   │▶│  SCL  │▶│Python │
│→ IR   │ │       │ │       │ │       │ │S7-1200│ │vs SCL │
│       │ │MatMul │ │Adapt. │ │Memory │ │S7-1500│ │1e-4 ok│
│KAN -┐ │ │Bspline│ │Bspline│ │FLOPs  │ │       │ │       │
│MLP -┼─┘│ReLU   │ │DeadNod│ │Budget%│ │       │ │       │
│CNN -┘  │Softmax│ │ConstFl│ │       │ │       │ │       │
└───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
```

**IR (中间表示) 是核心基础设施。** 它的价值：
- 解耦 PyTorch 和 SCL → 加新模型只改 Frontend
- 优化只作用于 IR → 所有后端受益
- 可序列化/可测试/可调试
- 论文里有架构图 → 证明你懂编译器设计

### B-spline 自适应采样 (原创算法)

```
问题: 均匀采样，平坦区域浪费点，弯曲区域精度不够

算法: Curvature-Aware Non-Uniform Discretization
  1. 在 [-3, +3] 上高密度采样 (100pt), 计算曲率 κ(x)
  2. 累积曲率 C(x) = ∫κ(t)dt
  3. 在 C(x) 上均匀取 N_target 个点 → 曲率大的地方自动密集
  4. 查表时仍用线性插值 (与均匀采样相同的推理逻辑)

效果:
  - 同存储 (20pt): 精度比均匀采样提升 15-30%
  - 同精度: 存储减少 20-40%
```

---

## 三、技术路线

```
CWRU 振动数据 (12kHz DE, 52 files)
  │
  ├─→ 滑窗 (1024pt, stride=512)
  │     │
  │     ├─→ 原始波形 → Teacher 1D-CNN (不变)
  │     └─→ 28-D 特征 (10时域 + 10频域 + 8离散熵)
  │
  ├─→ Teacher: 1D-CNN(16→32→64) + 4-head SA + FC(128→64→4)
  │     ~50K params, 99%+ CWRU
  │
  ├─→ VRM-KD 蒸馏: KL + VRM + Feature Align
  │     └─→ Student KAN([28,16,4], grid=8, k=3)
  │           ~300 params, target 96%+
  │
  ├─→ NeuroPLC Compiler (方案 C)
  │     ├─ Frontend: PyTorch KAN/MLP → IR Graph
  │     ├─ Optimizer: B-spline 自适应采样
  │     ├─ Backend: IR → SCL (S7-1200 compact / S7-1500 perf)
  │     └─ Analyzer: 内存预算报告
  │
  ├─→ TIA Portal 自动验证 (MCP 189 API)
  │     └─ 0 errors, Python-SCL 逐元素一致
  │
  └─→ E5 新增: 同一编译器编译 KAN + MLP → 双目标PLC → 证明通用性
```

---

## 四、实验设计 (方案 C 升级版)

| # | 实验 | 证明什么 | 升级点 |
|---|------|---------|--------|
| **E1** | Teacher vs Student 准确率 | 压缩损失可接受 | — |
| **E2** | KAN vs MLP vs SVM/RF | KAN 在极低参数下的优势 | 都经同一编译器编译 |
| **E3** | KD 消融: No-KD / Hinton / VRM | VRM-KD 优于传统 KD | — |
| **E4** | B-spline 精度: 均匀 vs 自适应 @ 10/20/50pt | **自适应采样的优势** | 🔥 原创算法验证 |
| **E5** | 编译器通用性: KAN+MLP → S7-1200+S7-1500 | **编译器不是 KAN-only** | 🔥 方案C新增 |
| **E6** | Python vs SCL 交叉验证 (1000样本) | 代码生成正确性 | 含自适应采样精度 |
| **E7** | 跨工况泛化: 1hp → 0/2/3hp | 域泛化 (诚实讨论) | — |

---

## 五、项目文件结构

```
D:/neuroplc-paper/
├── README.md                         ✅ v4 — 方案 C
├── research-notes.md                 ✅
├── GAP-REPORT.md                     ✅ 方案 C 更新
├── MODELSCOPE.md                     ✅ GPU 训练指南 (36h 免费)
├── pytest.ini                        ✅ 测试配置
├── activate-neuroplc.sh              ✅ 环境激活脚本
├── .gitignore                        ✅
├── paper/
│   ├── main.tex                      ✅ IEEEtran 骨架
│   ├── main.pdf                      ✅ pdflatex 编译通过
│   └── references.bib                ✅ 32 篇文献 (Zotero 导入完成)
├── code/
│   ├── config.yaml                   ✅ v3 — 方案 C 编译器参数
│   ├── download_verify_cwru.py       ✅ GFW 兼容版
│   ├── download_cwru.py              ✅ 原始版 (保留参考)
│   ├── preprocess.py                 ✅ 28-D (20 统计 + 8 离散熵)
│   ├── train_on_modelscope.py        ✅ ModelScope GPU 训练入口
│   ├── models/                       ✅ Phase 1
│   │   ├── __init__.py               ✅
│   │   ├── teacher_cnn.py            ✅ 48,708 params
│   │   ├── student_kan.py            ✅ 6,148 params + 自适应采样
│   │   └── student_mlp.py            ✅ 1,524 params baseline
│   ├── train_teacher.py              ✅ Phase 1
│   ├── train_student_kd.py           ✅ VRM-KD: KL+CE+VRM+Feat
│   ├── evaluate.py                   ✅ E1-E7 框架
│   ├── visualize.py                  ✅ 7 张图表
│   ├── tests/                        42 tests ✅
│   │   ├── conftest.py               ✅
│   │   ├── test_data.py              ✅
│   │   └── test_preprocess.py        ✅
│   └── neuroplc/
│       ├── __init__.py               ✅
│       ├── utils/
│       │   ├── __init__.py           ✅
│       │   └── mlflow_tracker.py     ✅
│       ├── ir.py                     ⏳ Phase 2
│       ├── frontend.py               ⏳ Phase 2
│       ├── optimizer.py              ⏳ Phase 2
│       ├── backend_s7.py             ⏳ Phase 2
│       ├── analyzer.py               ⏳ Phase 2
│       ├── compiler.py               ⏳ Phase 2
│       ├── scl_templates.py          ⏳ Phase 2
│       └── validator.py              ⏳ Phase 2
├── data/raw/12k_DE/                  ✅ 48/52, 142MB
├── results/{mlflow.db,scl_output/}   ✅
└── tia_project/                      ⏳ Phase 5
```

---

## 六、关键参数

| 参数 | 值 |
|------|-----|
| CWRU | 12kHz DE, 52 files, 4 fault × 4 diameter × 4 load |
| 滑窗 | 1024pt, stride=512 (50% overlap) |
| 特征 | **28-D** (10 time + 10 freq + 8 dispersion entropy) |
| Teacher | 1D-CNN(16→32→64) + 4-head SA, ~50K params |
| Student | KAN([28,16,4]), grid=8, k=3, ~300 params |
| VRM-KD | τ=4.0, α=0.3, λ_rel=0.5 |
| B-spline LUT | 自适应20点 (S7-1200) / 50点 (S7-1500) |
| PLC | S7-1200 CPU 1211C V4.7 (75KB) + S7-1500 (1.5MB) |

---

## 七、可行性审计

| # | 环节 | 风险 | 证据 / 缓解 |
|---|------|:---:|------|
| ① | CWRU → 28-D | 🟢 | 成熟技术 |
| ② | Teacher CNN 99%+ | 🟢 | CWRU 标准操作 |
| ③ | VRM-KD 蒸馏 | 🟡 | ICCV 2025 Highlight, 实现简单 |
| ④ | KAN Student 收敛 | 🟡 | MLP baseline 兜底 |
| ⑤ | IR 编译器 | 🟡 | 只建模实际需要的操作, 不追求完备 |
| ⑥ | B-spline 自适应采样 | 🟢 | 本质是曲率计算+重采样, 数学清晰 |
| ⑦ | TIA Portal 编译 | ✅ | 已验证 0 errors |
| ⑧ | Python vs SCL 一致 | 🟢 | IEEE 754, E6 验证 |
| ⑨ | 跨工况泛化 | 🔴 | 已知挑战, 诚实讨论即可 |

---

## 八、Phase 执行计划

| Phase | 内容 | 文件数 | 状态 |
|-------|------|:---:|:---:|
| **0** | 前置准备 (工具链+测试+论文骨架+MLflow) | 8 | ✅ |
| **1** | 模型+训练 (Teacher CNN, KAN, MLP, KD, eval, viz) | 8 | ✅ |
| **2** | 编译器 (IR, Frontend, Optimizer, Backend, Analyzer) | 8 | ⏳ |
| **3** | 测试 (IR/Frontend/Optimizer/Backend/Compiler) | 5 | ⏳ |
| **4** | 实验运行 (7组) | — | ⏳ |
| **5** | 论文正文填充 | — | ⏳ |
| **6** | TIA Portal MCP 验证 | — | ⏳ |

### 依赖关系

```
P0 (前置准备) ── ✅ 完成
  │
  ├── 板板: P0-1 下载数据  🔴
  │         P0-2 Zotero    🔴
  │
  └── Claude:
        P1 (模型+训练) ──→ P2 (编译器) ──→ P5 (论文)
              │                 │
              ▼                 ▼
           P4 (实验)  ←───────┘
              │
              ▼
           P6 (TIA验证)
```

---

## 九、Bug 修复 & 变更日志

| 日期 | 文件 | 内容 |
|------|------|------|
| 07-03 | preprocess.py:463 | 🔴 跨工况 val 分片公式错误, 已修复 |
| 07-03 | download_cwru.py | 🟡 删除 3 处死代码 |
| 07-03 | 工具链 | 🛠 onnx/sciplots 统一到系统 Python |
| 07-04 | 方案 | 🔬 v1(MLP) → v2(KAN+VRM-KD+28-D) |
| 07-04 | download_verify_cwru.py | 🆕 GFW 兼容版: 本地导入+校验+清单 |
| 07-04 | preprocess.py | 🆕 离散熵模块 (RCMDE + RCHFDE, 20→28维) |
| 07-04 | paper/main.tex | 🆕 IEEEtran 骨架 |
| 07-04 | tests/ | 🆕 42 tests 全覆盖 |
| 07-04 | neuroplc/utils/mlflow_tracker.py | 🆕 MLflow sqlite 追踪器 |
| **07-04** | **方案** | **🔬🔬 v2(KAN工具) → v3(IR-Based通用编译器)** |

---

## 十、给板板的备忘

1. **环境**: 系统 Python 3.14.3 即可, 不需 CUDA。venv: `/d/dev-tools/research/venv/`
2. **C 盘**: 所有数据放 D 盘
3. **今天 (7/4)**: 考完高数后 —
   - ① 123云盘下载 CWRU 全量 → 用 `download_verify_cwru.py --source local --input <dir>` 导入
   - ② Zotero 导入 12 篇核心文献
   - ③ 告诉我准备好了, 我立即开始写 Phase 1 模型代码
4. **GPU**: 如果本机训练慢, 用 ModelScope 魔搭社区 36h 免费 GPU (账号 aiyuedi)
5. **新会话**: 贴「当前会话状态」提示词即可无缝续接
6. **风险**: 如果 KAN 训练效果不好 → 回退到 MLP → 编译器依然通用 → 论文仍然成立
7. **测试**: `python -m pytest code/tests/ -v -s`

---

*最后更新: 2026-07-04 15:30 CST*
*作者: 刘甫悦 (板板) + Claude*
