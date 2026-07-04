# NeuroPLC — 下一阶段打磨指南

> 写给下一个聊天框的 Claude，也是写给板板的备忘录。
> 当前状态: 论文第一版草稿，数据真实，编译通过，距"研究生顶尖水平"还需打磨。
> 日期: 2026-07-04 23:00 CST

---

## 快速自检（运行这条确认一切正常）

```powershell
cd D:\neuroplc-paper\code
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONPATH="D:\neuroplc-paper\code"
D:\dev-tools\research\venv\Scripts\python.exe -m pytest tests/ -q
```

预期: `73 passed`

---

## 当前论文的不足（按优先级排序）

### P0: Figure 嵌入 + 引用（30 min）

paper/main.tex 目前只嵌入了 2 张图（Fig 1 overview + Fig 2 compiler_arch）。
还有 5 张已经生成好的图在 `results/figures/`，需要：

| 文件名 | 论文用途 | 放置位置 |
|--------|---------|---------|
| `fig3_bspline_adaptive.pdf` | 均匀 vs 自适应采样对比 | E4 实验段落 |
| `fig4_kan_activations.pdf` | 学到的 B-spline 激活函数 | 方法论 (§3.2.3) 或 E1 |
| `fig5_confusion_matrices.pdf` | Teacher + Student 混淆矩阵 | E1 实验段落 |
| `fig6_tsne_features.pdf` | No-KD/Hinton-KD/VRM-KD t-SNE | E3 实验段落 |
| `fig7_cross_validation.pdf` | Python vs SCL 误差分布 | E6 实验段落 |

**做法:** 把 pdf 从 `results/figures/` 复制到 `paper/figures/`，在 `main.tex` 对应位置加 `\includegraphics`。

### P1: 补 Results Summary 表（15 min）

现在 E1-E7 的文字描述分散在各段落，缺一张总览表。
建议在 Experiments 章节末尾加一张表：

```
| Exp | Description | Key Metric | Result |
| E1  | Teacher vs Student | Acc | 99.98% vs 99.99% |
| E2  | KAN vs MLP vs SVM/RF | Acc | All 100% |
| E3  | KD Ablation | VRM/Hinton/No-KD | 99.99/99.93/24.13% |
| E4  | LUT Precision | 10/20/50pt acc | All 100% |
| E5  | Compiler Generality | 4/4 targets | All pass |
| E6  | Cross-Validation | MaxAE/RMSE | 0/0 |
| E7  | Cross-Load | 0/2/3hp acc | 99.97/100/99.97% |
```

### P2: 方法论精简（45 min）

现在 §3 太啰嗦。压缩方向：
- §3.2 编译器架构 → 保留 IR/前端/后端核心，删过于细节的描述
- Algorithm 1（B-spline LUT）已够，不要再加伪码
- 自适应采样算法 → 强调"原创性"，加一组对比数字(均匀 vs 自适应误差)
- Feature 工程和 Teacher/Student 训练 → 各压缩成一段

### P3: Introduction 重构（30 min）

现在 intro 偏"survey"。改成经典三幕结构：
1. **Problem** (1段): PLC 部署神经网络的三个障碍
2. **Why KAN** (1段): 参数量少 + B-spline 天然可离散化
3. **Gap** (1段): 没有工具把 KAN 编译到 IEC 61131-3
4. **Our Solution** (1段): NeuroPLC 编译器
5. **Contributions** (bullet points, 已经写得不错)

### P4: 润色检查清单（1h）

- [ ] 所有 `\ref{}` 都解析正确（编译后看 main.log 有 no undefined refs）
- [ ] 所有 `\cite{}` 都在 references.bib 里有对应条目
- [ ] 表格数字有效位数统一（都是 2 位小数 or 4 位小数，不要混）
- [ ] 图注完整（每张图有描述性 caption）
- [ ] 没有"will be updated"或"to be done"之类的占位符
- [ ] Section 标题格式一致
- [ ] 检查是否有 orphan/widow 行（单行跨页）

### P5: TIA Portal 真实验证（可选，但会大幅加分）

如果板板本机有 TIA Portal V21 + Openness:
```bash
# 用 MCP 自动验证编译器生成的 SCL 代码
# 参考: 记忆 tia-v21-mcp-success
```

---

## 关键文件位置

```
D:\neuroplc-paper\
├── paper/
│   ├── main.tex          ← 论文主文件（需打磨）
│   ├── main.pdf          ← 当前编译结果 (6页, 326KB)
│   ├── references.bib    ← 32 篇文献
│   └── figures/          ← 论文图片放这里
├── results/
│   ├── evaluation/
│   │   └── evaluation_results.json  ← E1-E7 真实数据
│   ├── figures/          ← 7 张已生成的图 (PDF+PNG)
│   ├── scl_output/       ← 编译器生成的 SCL 代码
│   ├── teacher/          ← Teacher CNN checkpoint
│   └── student/          ← 4 个 Student checkpoint
├── code/
│   ├── neuroplc/         ← 编译器 8 模块
│   ├── models/           ← 3 个模型定义
│   ├── train_teacher.py  ← Teacher 训练
│   ├── train_student_kd.py ← Student KD 训练
│   ├── evaluate.py       ← E1-E7 评估
│   ├── visualize.py      ← 7 图生成
│   └── tests/            ← 73 个测试
├── GAP-REPORT.md         ← 项目进度总览
├── NEXT-STEPS.md         ← 本文件
└── run_all.sh            ← 一键跑通全流程
```

---

## 实验数据速查（论文需要的数字全在这）

**训练结果:**
| 模型 | Test Acc | 参数量 | 时间 |
|------|:---:|:---:|:---:|
| Teacher CNN | 99.98% | 48,708 | 116s |
| KAN VRM-KD | 99.99% | 6,148 | 58s |
| KAN Hinton-KD | 99.93% | 6,148 | 53s |
| KAN No-KD | 24.13% | 6,148 | 42s |
| MLP VRM-KD | 99.89% | 1,524 | 37s |

**编译器结果:**
| 目标 | IR节点 | 内存(KB) | 预算(%) |
|------|:---:|:---:|:---:|
| KAN → S7-1200 | 11 | 40.3 | 53.7 |
| KAN → S7-1500 | 11 | 110.8 | 7.4 |
| MLP → S7-1200 | 8 | 13.2 | 17.6 |
| MLP → S7-1500 | 8 | 13.2 | 0.9 |

**E6 交叉验证:** MaxAE=0, MAE=0, RMSE=0, 100%一致
**E7 跨负载:** 0hp 99.97% / 2hp 100% / 3hp 99.97%

---

## 编译论文命令

```bash
cd D:/neuroplc-paper/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## 对下一个 Claude 的提示

1. 先读 `GAP-REPORT.md` 了解整体架构
2. 再读 `paper/main.tex` 了解论文现状
3. 按 P0→P1→P2→P3→P4 顺序打磨
4. 每次修改后 `pdflatex` 验证编译通过
5. 板板（刘甫悦，桂电智能制造大二，称呼"板板"）偏好结构化和可执行的方案
6. 板板本机无 GPU（CPU-only PyTorch），但模型小够用
7. 代码已推送 Gitee: `https://gitee.com/aiyue-emperor/neuroplc`
8. mlflow 已惰性导入，没装也不炸
