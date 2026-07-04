# NeuroPLC 文献调研笔记 — 如何同时提升算法和系统层次

> 调研日期: 2026-07-03
> 目标: 找到能让小论文「既有优秀算法，又有优秀系统」的方案

---

## 一、四大候选前沿架构分析

### 1. KAN (Kolmogorov-Arnold Networks) ⭐⭐⭐⭐⭐ 强烈推荐

**核心思想:** 把传统 MLP 的固定激活函数(ReLU)替换为可学习的 B-spline 激活函数。
每个 "权重" 不是标量，而是一个小函数。

**最新证据 (2024-2025):**
| 论文 | 数据集 | 准确率 | 参数量 | 亮点 |
|------|--------|--------|--------|------|
| Rigas et al. (Entropy 2025) | CWRU + MaFaulDa | **100% F1** | 极少 | 开源代码 GitHub, 自动特征选择, 符号公式可解释 |
| HKAC (IEEE 2025) | 航空发动机轴承 | 高于MLP | **仅35% MLP参数量** | KAN替换MLP分类头 |
| LQCKAN (NDT.net 2025) | CWRU | **99.99%** | 轻量 | QCNN+CKAN双流+动态剪枝 |
| FastKAN-DDD (PLoS ONE 2025) | TinyML | 99.94% | **35KB, 0.04ms** | 证明KAN适合超轻量部署 |

**为什么适合你的项目:**
- KAN 的可学习激活函数本质是 B-spline = 分段多项式
- 在 SCL 中可以实现为: 对每段区间用多项式求值 → 完全可以编译!
- 参数效率远超 MLP (同样的参数量, KAN 表达能力更强)
- KAN 自带**可解释性**: 学到的激活函数可以写成数学公式, 论文里可以展示

**与 PLC SCL 的天然适配:**
```
B-spline 激活函数 = 分段三次多项式
在 SCL 中 = CASE OF 区间判断 + 多项式求值
或者 = 查表法 (10点查表 + 线性插值, 精度损失 < 0.1%)
```

### 2. Mamba (State Space Models) ⭐⭐⭐⭐ 推荐但实现复杂

**核心思想:** 线性复杂度的序列建模, 替代 Transformer 的 O(n²) 注意力。

**最新证据 (2025):**
| 论文 | 参数 | 准确率 | 亮点 |
|------|------|--------|------|
| WCamba (AEI 2025) | **0.016M** | 95.44% | 极噪声条件下, 推理加速52% |
| VibrMamba (Measurement 2025) | 极少 | 99.77-99.95% | **Mamba + KAN 混合**, 4个数据集 |
| PG-TMT (arXiv 2025) | 轻量 | 跨工况强 | 物理引导 + Tiny-Mamba + Transformer |
| TFG-Mamba (AEI 2025) | **0.081M** | 优秀 | RUL预测, 比Transformer少49%参数 |

**问题:** Mamba 的 selective scan 机制依赖状态变量 + 门控 + 离散化, 在 SCL 中实现非常复杂。

### 3. Physics-Informed Lightweight Models ⭐⭐⭐ 可以借鉴思路

**核心思想:** 把轴承物理知识(故障特征频率、动力学方程)嵌入模型结构或输入。

**代表:**
- PLT-Bearing (2025): patch 大小由故障频率公式决定 → 跨工况零样本 90.3%
- PI-LSTM (2025): 输入加入从动力学方程推导的速度和力 → +6-9% 准确率
- BWKNet (2025): 第一层用双阻尼小波核(匹配轴承脉冲响应) → 99.86%

### 4. 高级 Knowledge Distillation ⭐⭐⭐ 可以替换 Hinton KD

**最新方法 (2024-2025):**
- VRM (Virtual Relation Matching, ICCV 2025): 虚拟关系匹配 → 比传统KD好
- 多阶段剪枝-蒸馏交错 (Ren 2024): 剪枝→蒸馏→再剪枝→再蒸馏, 部署Jetson Nano
- Feature-level KD + 2.68KB模型 + FPGA: 证明了极低参数量下KD的有效性

---

## 二、特征工程升级方案

### 当前方案 (20-D):
10 时域 (RMS/峰值/峭度/...) + 10 频域 (频谱质心/频带能量/...)

### 可升级方向: Dispersion Entropy 族 (2024-2025 主流)

| 方法 | 年 | 特点 |
|------|-----|------|
| HRCGMFDE | 2024 | 同时提取低频+高频故障信息 |
| RCHFDE | 2024 (IEEE Sensors) | 模糊离散熵 + 多级去噪 |
| RCMPNDE | 2025 | 峰峰值归一化, 98.5%准确率 |
| RCMREDE 2D | 2025 | 2D雪花图熵, 捕捉幅值+频率 |

**可行方案:** 在当前 20-D 特征的基础上加 5-10 维多尺度离散熵特征, 不改变模型架构, 提升特征表达力。

---

## 三、编译器/代码生成领域确认

### 关键发现: 你的方向确实是空白

| 工具 | 做什么 | 与你的关系 |
|------|--------|-----------|
| Agents4PLC (ZJU 2024) | LLM 多智能体生成 ST 代码 | 用LLM写PLC逻辑, 不是ML→PLC |
| LLM4PLC (IEEE 2024) | LLM + nuXmv 形式化验证 | 同上 |
| PyLC+ (瑞典 2025) | PLC代码→Python翻译+验证 | 反向操作(PLC→Python) |
| **NeuroPLC (你)** | **PyTorch→SCL自动化** | **无竞争对手** |

---

## 四、最终推荐方案: KAN + 增强 KD + NeuroPLC 编译器

### 为什么选 KAN 而不是 Mamba/CNN/MLP

| 标准 | KAN | Mamba | CNN (depthwise) | MLP |
|------|:---:|:---:|:---:|:---:|
| 算法新颖度 (2024-2025) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| 参数效率 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| SCL 可编译性 | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 可解释性 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ |
| 开源代码可用 | ✅ GitHub | ✅ | ✅ | ✅ |
| 适合小论文 | ✅ | 实现太复杂 | ✅ | 太普通 |

### 改进后的技术路线

```
CWRU振动数据
  ↓
特征提取 (升级版: 20-D统计 + 8-D离散熵 = 28-D)
  ↓
Teacher: 1D-CNN (不变, 标准大模型, 99%+)
  ↓
Knowledge Distillation (升级: 关系匹配KD + 温度蒸馏)
  ↓
Student: Shallow KAN (2层, ~300参数, 目标95%+)
  ↓
NeuroPLC Compiler (升级: 支持KAN的B-spline查表编译)
  ↓
TIA Portal 编译验证 (0 errors ✓)
  ↓
HMI 诊断仪表盘
```

### 论文中的创新陈述将变成:

> **算法创新 (2项):**
> 1. 首次将 KAN 应用于 PLC 可部署的轴承故障诊断 — B-spline 可学习激活函数在保持极高参数效率的同时提供可解释性
> 2. 结合关系匹配知识蒸馏 (VRM-KD) 和多尺度离散熵特征, 在 <500 参数下实现 >95% 准确率
>
> **系统创新 (2项):**
> 1. 设计并实现了 NeuroPLC — 首个支持 KAN 架构的 PyTorch → IEC 61131-3 SCL 自动编译器
> 2. 通过 TIA Portal MCP (189 API) 实现全自动编译验证 + 端到端 HMI 诊断系统

---

## 五、参考文献速查 (推荐的 12 篇核心引用)

### KAN 相关
1. Rigas et al. "Explainable fault and severity classification for rolling element bearings using Kolmogorov-Arnold networks." Entropy, 2025.
2. Li et al. "Fault Detection and Diagnosis... Using Time-Frequency Domain Filters and CNN-KAN." Systems, 2025.
3. HKAC: "Aero-engine Inter-shaft Bearing Fault Diagnosis via Hybrid Kolmogorov-Arnold Classifier." IEEE, 2025.
4. Liu et al. "KAN: Kolmogorov-Arnold Networks." (原始 KAN 论文, arXiv 2024)

### Mamba/轻量架构
5. VibrMamba: "A lightweight Mamba based fault diagnosis of rotating machinery." Measurement, 2025.
6. WCamba: "Lightweight fault diagnosis for aero-engine via wide-kernel convolution and state space modeling." AEI, 2025.

### 知识蒸馏
7. Zhang et al. "VRM: Knowledge Distillation via Virtual Relation Matching." ICCV, 2025.
8. Ren et al. "Lightweight Intelligent Fault Diagnosis Method Based on Multi-Stage Pruning Distillation." Advances in Mechanical Engineering, 2024.

### 特征工程
9. Ding et al. "Two-dimensional refined composite multi-scale revised ensemble dispersion entropy." EAAI, 2025.
10. Chen et al. "HRCGMFDE: Hierarchical Refined Composite Generalized Multiscale Fluctuation Dispersion Entropy." Shock & Vibration, 2024.

### PLC 代码生成
11. Liu et al. "Agents4PLC: Automating Closed-loop PLC Code Generation and Verification." arXiv, 2024.
12. Fakih et al. "LLM4PLC: Harnessing Large Language Models for Verifiable Programming of PLCs." IEEE, 2024.

---

## 六、量化评估

| 指标 | 当前方案 (MLP) | 升级方案 (KAN) |
|------|:---:|:---:|
| 算法新颖度 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 系统新颖度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 模型参数 | ~1268 | ~300-500 |
| 准确率预期 | ~95% | ~97%+ |
| SCL 编译复杂度 | 低 (矩阵乘法) | 中 (B-spline查表) |
| 论文可解释性 | 无 | 符号激活函数 |
| 导师/研究生反应 | "还行" | "KAN是什么? 你居然编译到PLC了?" |
| 实现风险 | 低 | 中 (需学习KAN + B-spline编译) |
