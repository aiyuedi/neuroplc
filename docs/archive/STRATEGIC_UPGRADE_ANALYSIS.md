# NeuroPLC 论文深度提升战略分析

> 2026-07-09 | 基于全网文献搜索 + 论文全文审读

---

## 一、当前定位与竞争格局分析

### 1.1 论文在领域的独特贡献 (已验证, 无需改动)

| 维度 | NeuroPLC 的独占贡献 | 竞争状态 |
|------|-------------------|---------|
| PyTorch→IEC 61131-3 SCL 编译器 | **唯一** | RTNNIgen (Keras→ST, 无验证), MLconverter (sklearn→ST, 无验证) |
| 设计时正确性证书 | **唯一** | 所有竞争者仅提供测试集经验验证 |
| SVNN 形式框架 (可编译前沿) | **独创** | 无类似理论 |
| Doubleton Arithmetic (DA) | **独创** (在编译验证上下文中) | Krukowski 2024 "Make IBP Great Again" 用 Doubleton 做 NN 验证，但面向 ReLU |
| B-spline LUT 量化编译 | 有竞争但差异化 | KANELÉ (FPGA, ISFPGA 2026 Best Paper), LUT-KAN (MCU, Neurocomputing 2026) |

### 1.2 竞争威胁评估

#### 🔴 高威胁: KANELÉ (MIT, ISFPGA 2026 Best Paper)
- **做的**: 首个系统化 KAN→FPGA LUT 部署框架，2700× 加速
- **与 NeuroPLC 重合**: 都利用了 "KAN B-spline 天然适合 LUT 离散化"
- **NeuroPLC 的差异化点**: (1) 目标平台不同 (PLC vs FPGA); (2) **形式正确性证明 vs 经验验证; (3) SVNN 理论框架**
- **行动**: 论文已正确引用 KANELÉ 并差异化 → **无需改动**

#### 🟡 中威胁: QuantKAN (Fuad & Chen, 2025)
- **做的**: KAN 统一量化框架 (QAT+PTQ)，分支专用量化器
- **可操作项**: 论文目前未引用 QuantKAN。**必须引用**并说明 NeuroPLC 的 LUT 方法是一种特殊的 KAN 量化策略（面向 IEC 61131-3 REAL 32-bit）

#### 🟡 中威胁: LUT-KAN / SHARe-KAN
- LUT-KAN (Kuznetsov, Neurocomputing 2026): B-spline→LUT 转换, 7-25× 加速
- SHARe-KAN (Smith, 2025): 向量量化, 9.3× 压缩
- **可操作项**: 论文已引用 LUT-KAN，但应更清晰区分: NeuroPLC 不是做 LUT 加速，而是做 LUT **编译正确性证明**

#### 🟢 低威胁: LLM4PLC / Agents4PLC / AutoPLC
- 全是用 LLM 生成 PLC 代码，不是神经网络编译器
- 对 NeuroPLC 论文的帮助: 可在 Related Work 提及作为正交方向

#### 🟢 低威胁: ESBMC-PLC+ (2026)
- SMT-based 形式验证 IEC 61131-3 程序
- 与 NeuroPLC 互补: 可组合使用

---

## 二、论文当前弱点诊断 (按严重度排序)

### 🔴 致命弱点 (必须修复才能投 Q1)

#### 2.1 缺失关键竞争引用: QuantKAN + SHARe-KAN
- **QuantKAN** (arXiv 2511.18689): 第一个 KAN 统一量化框架
- **SHARe-KAN** (arXiv 2512.15742): 后训练向量量化，9.3× 压缩
- **影响**: Reviewer 会问 "KAN 量化文献为什么不引用？"

#### 2.2 DA 的原始创新性需要更清晰声明
- Krukowski 2024 "Make IBP Great Again" 已经用了 Doubleton Arithmetic 做 NN 验证
- **当前论文已引用此文献** (krukowski2024doubleton)，但需要更明确区分:
  - Krukowski 的 DA 用于 ReLU 网络、面向通用验证
  - NeuroPLC 的 DA 用于 **B-spline KAN、面向编译到 SCL 的误差传播**
  - 关键差异: **符号结构分析** (sign-structural) + **段精确性** (segment-exactness) 是原创

#### 2.3 缺少数值仿真/PLCSIM 在线验证
- 论文声称 "design-time correctness" 但从未在真实 PLC 上验证
- **可以做的**: 用 PLCSIM Advanced (已有 Python ctypes bridge) 做在线对比验证
  - Python float32 vs PLCSIM REAL 逐输出对比
  - 验证 DA 计算的 bound 确实覆盖实际误差
- **这不需要物理硬件!** PLCSIM Advanced 是纯软件仿真器

#### 2.4 缺乏与 KANELÉ 的定量对比
- 定性对比已做 (ISFPGA Best Paper vs our PLC)，但缺少:
  - 同一模型在 FPGA(KANELÉ) vs PLC(NeuroPLC) 的推理延迟/内存对比
  - 如果 PLCSIM 做不到，至少需要做一个推算对比表

### 🟡 中等弱点 (影响论文竞争力)

#### 2.5 SVNN 框架的理论深度可以进一步加强
- 当前: 7 theorems + 8 propositions
- **可加强**: 
  - 将 SVNN 条件与 **abstract interpretation 的 Galois connection** 形式关联
  - 已有 Cousot 1977 引用但未充分展开
  - 证明 DA 抽象域在 B-spline 函数类上的 **完备性** (completeness)

#### 2.6 实验数据集太传统
- CWRU (2015年前的轴承台架)
- XJTU-SY (2020, 也是轴承)
- **可以加**: 一个新的公开工业数据集
  - Paderborn (已有引用 lessmeier2016paderborn)
  - 或 PU 轴承数据集
  - 或一个非轴承工业数据集

#### 2.7 MNIST 实验可能被挑战为 "不相关"
- MNIST 不是工业场景
- 如果要用 MNIST，需要更强调它是 "标准化可复现验证" 而非声称工业应用

### 🟢 次要改进

#### 2.8 DA/IA 段内 2.82× vs 端到端 2.2× 的解释已添加，可以更量化
#### 2.9 缺少对 PLC 扫描周期约束下的推理延迟分析
#### 2.10 中文版本未同步 (用户已说先不管)

---

## 三、可执行的提升方案 (按投入产出比排序)

### 方案 A: PLCSIM 在线验证 (投入中等, 回报极高) ⭐⭐⭐⭐⭐

**目标**: 在 PLCSIM Advanced 中运行编译后的 SCL，与 Python 逐输出对比

**执行步骤**:
1. 导出 KAN 模型的 SCL (已可做)
2. 用已有的 PLCSIM Python bridge 下载到虚拟 PLC
3. 写入 100 个测试输入 → 读取输出 → 对比 Python float32
4. 计算每个输出的误差，验证最大值 ≤ DA bound

**产出**:
- 一个新实验 (E59)
- 一个表格 (100 测试样本, 4 个输出, 实测 vs DA bound vs IA bound)
- 一个图 (误差分布直方图)
- **最强证据**: "我们的 compiler 声称的 bound 在真实 PLC 上成立"

**文件**: `code/experiments/e59_plcsim_verification.py`

### 方案 B: 引用补全 + Related Work 强化 (投入低, 回报高) ⭐⭐⭐⭐

**需要新增引用**:
1. **QuantKAN** (arXiv 2511.18689) — KAN 量化框架
2. **SHARe-KAN** (arXiv 2512.15742) — 向量量化压缩
3. **ESBMC-PLC+** (arXiv 2606.15461, 已有?) — 需要确认是否已引用
4. **Agents4PLC** — 多智能体 PLC 代码生成 (显示 LLM 方法不能做编译验证)
5. **LUT-KAN (DoS detection)** (arXiv 2601.08044) — LUT 编译用于 IoT 边缘

**执行**: 在 Related Work 和 Discussion 中新增 1-2 段

### 方案 C: DA 完备性理论深化 (投入中等, 回报高) ⭐⭐⭐⭐

**目标**: 用 MATLAB 符号数学证明 DA 是 B-spline 函数类上的 **最紧** 抽象 (Galois connection 完备性)

**核心声明**: 对于单变量分段多项式 (B-spline segment)，DA 是 **完备的** (complete abstract interpretation): 
- DA 的抽象函数 α 和具体化函数 γ 形成 Galois 连接
- DA 在仿射段上是 **精确的** (exact) → 已证明 (Prop 4.1)
- 在非仿射段上 O(r²) 是 **最优的** (optimal) → 需要证明: 对于度≥2 的多项式，任何基于一阶 Taylor 的仿射抽象都不可能有比 O(r²) 更紧的界

**产出**: 一个 Proposition/Theorem + MATLAB 符号证明

### 方案 D: 与 KANELÉ 定量对比 (投入低, 回报中等) ⭐⭐⭐

**目标**: 在一张表中对比 NeuroPLC vs KANELÉ 的关键指标

| 指标 | NeuroPLC (S7-1200) | KANELÉ (FPGA) |
|------|-------------------|---------------|
| 目标平台 | Siemens PLC | Xilinx FPGA |
| 目标语言 | IEC 61131-3 SCL | VHDL RTL |
| LUT 大小 | N=15 (60 floats/edge) | 可配置 |
| 形式正确性证书 | ✅ SVNN + DA | ❌ 仅经验验证 |
| 编译到目标 | ✅ TIA Portal 0e0w | ❓ |
| 延迟 | <2.86ms (WCET Z3) | <100ns |
| 内存 | 45.2 KB | N/A |

### 方案 E: 第三个数据集 (投入低, 回报中等) ⭐⭐⭐

**推荐**: Paderborn 轴承数据集 (已有引用) 或 SEU 齿轮箱数据集

**只需**: 在现有代码上跑一次 fine-tune → 报告准确率 + 验证 Z3 保持率

---

## 四、优先级排序与预估时间

| 排序 | 方案 | 预计耗时 | 预期效果 | 阻塞物 |
|------|------|---------|---------|--------|
| 1 | **方案 B: 引用补全** | 1-2h | 消除 reviewer 最可能的质问 | 无 |
| 2 | **方案 A: PLCSIM 验证** | 3-5h | 最强经验证据 | 需要 PLCSIM Adv 环境 |
| 3 | **方案 C: DA 完备性理论** | 3-5h | 理论深度质的飞跃 | 无 (MATLAB 可用) |
| 4 | **方案 D: KANELÉ 定量对比表** | 1h | 强化差异化 | 无 |
| 5 | **方案 E: 第三个数据集** | 2-3h | 提升实验完备性 | 需要下载数据集 |

---

## 五、潜在的致命问题 (需要诚实面对)

### 5.1 "设计时正确性" 与 "浮点误差" 之间的 gap
- Szász et al. (ICML 2025 Spotlight) 证明: IEEE 754 浮点误差可能产生验证器看不到的后门
- NeuroPLC 的 DA 基于 **实数域** 计算，但 SCL 运行在 REAL (32-bit IEEE 754)
- **论文已处理**: 通过安全裕度 (≥6×) 覆盖浮点舍入
- **但可以更强**: 用 PLCSIM 实测证明安全裕度足够

### 5.2 98.6% MNIST 可能被 reviewer 质疑
- "为什么要用 MNIST？这不是工业场景"
- 论文已解释为 "标准化可复现性验证"
- 建议在正文中更强调这一点

### 5.3 "IR Minimality" Lemma 的严谨性
- 声称 6-op IR 是 minimal 的需要更严谨的证明
- 当前是 proof sketch
- 可以扩展为正式 Lemma

---

## 六、最终建议

**立即执行 (今天)**:
1. 方案 B (引用补全) — 1-2h, 消除最大风险
2. 方案 D (KANELÉ 对比表) — 1h, 低投入

**优先执行 (这次改版)**:
3. 方案 A (PLCSIM 验证) — 如果你有 PLCSIM Advanced 环境
4. 方案 C (DA 完备性) — 如果 MATLAB 可用

**可选的锦上添花**:
5. 方案 E (第三个数据集)
