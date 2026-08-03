# 可认证部署容量理论 — 设计文档 v4（对抗复查修补版）

> 生效日：2026-08-04 | v4 变更：对抗性文献复查（lit-scout-capacity, 二轮）结论落地——领域划分成立，修补三漏洞（CP 先例、解耦措辞、指数声明）+ 三条红线
> 状态：P1 完成（设计定稿 + necessity 链闭合 + 对抗复查通过）；下一步 P2 数值验证

---

## 0. v3 → v4 变更记录（对抗复查的三漏洞与修补）

复查判定：**v3 验证侧领域划分成立**（"类级容量/层化理论"层无先例），但有三个审稿人必打点：

1. **漏洞 1（最致命）：CP 的"无先例"表述不成立**——Gappa（Daumas-Melquiond 2009）/ Sollya（2010）/ LeanCert（Lean 4）就是"部署界 + 算术/余项/区间规则 + 绝对 soundness + 机器可检证书"的完整先例。数值分析审稿人会写 "this is Gappa with a budget"。
   **修补**：CP 重定位为 **"Gappa/LeanCert 式证书演算 + 容量层化"**；related work 起点 = Gappa/Sollya/LeanCert/PCC/PCA/credible compilation/translation validation；差异化 = 三点（类级 packing 耦合、V 为预算的容量理论、necessity 定理）——**"构件有先例、理论无先例"句式**（红线②）。
2. **漏洞 2：V_B 解耦措辞会被秒杀**——SNARK/GKR/PCP 的简洁性是三大领域核心教义，审稿人会写 "you rediscovered succinctness"。
   **修补**：防御三件套 (i) **确定性 + 完美 soundness ⟹ V ≥ |π|**（PCP[0,poly(n)]=NP 定理——完美 soundness 下验证者必须读全证明；PLC 无随机源/无交互协议 ⟹ 确定性验证器，工业动机）；(ii) 对象 = 实数域类级语义界，非 F_p 电路计算（zkML 验证的是"承诺模型下正确计算"，非"输出对函数类的语义 ε-界"——对象不同）；(iii) 贡献 = 容量函数本体（C(B,V) 与层化），不是"验证比复算便宜"这一事实。
3. **漏洞 3：定理 3 的"结构最优性 ⟹ 指数验证"不能作一般原则**——SOS 精确有理证书（Kaltofen-Li-Yang-Zhi）证明多项式问题的全局最优性可多项式验证。
   **修补**：指数声明限定为**类级**（遍历所有节点集的结构最优 = 类级最优 ⟹ 指数，Thm 5 显式归约）；**实例级**（给定节点集的误差界）闭式可算——明确承认。Henzinger-Saraç（LICS 2021，资源-精度严格单调）引为精神祖先并区分（在线监控属性精度 vs 离线类级证明长度）。

**支持性材料（免费收获，写入论文）**：PCP[0,poly]=NP 支持 V=|π| 语义；Sistla-Clarke 1985（LTL MC PSPACE-complete，验证成本随规格增长）支持层 1"随 B 不增长"的稀缺性；GKR（无条件 soundness 交互式）反衬确定性模型；Buss-Kapron（STOC 2002，证书-验证指数分离）necessity 限定设定；CROWN/DeepPoly 闭式传播 = 层 1 机制的已有形态（不完备性谱 IBP<CROWN<LP<MILP = 层化支持材料）。

---

## 1. 形式设置（v4 定稿）

**对象澄清**（延续 v3）：激活级（已知函数逼近，T-I）与映射级（类内未知，容量层化）分离。

**CP 证明系统（v4 重定位——"Gappa 式证书演算 + 容量层化"）**：
- 规则 = {①算术 ②插值余项（T-I）③组合传播 ④谱尾界 ⑤比较}；证明项 = 界恒等式推导；验证 = 逐规则局部检查，V = 证明长度。
- **与 Gappa/Sollya/LeanCert 的差异（三类，每类一句话）**：(i) 他们处理**单函数**实例界（浮点库/函数逼近），我们处理**类级**界（packing/度量熵耦合——界是类的函数，非单实例）；(ii) 他们无**容量/层化**（V 不是预算、无层、无 universal 线）；(iii) 我们提供**necessity 定理**（解耦 ⟹ 闭式），他们只有正向演算。
- **验证器语义（v4 补强）**：确定性（PLC 无随机源/交互协议——工业动机）；完美 soundness（数学恒等式）；**PCP[0,poly(n)] = NP ⟹ V ≥ |π|**（完美 soundness + 确定性验证者必须读全证明——V = |π| 的语义辩护）。

**验证成本双维**（延续 v3）：V_B（随存储预算 B 的增长）+ V_arch（随架构规模）。层 1 = V_B 解耦 ∧ V_arch poly。**"解耦"措辞纪律**：不声称"验证比复算便宜"（教义级事实），声称"验证成本与存储预算的**增长方式**是容量理论的参数维度"（贡献 = 容量函数本体）。

---

## 2. 三定理（v4 定稿）

### 定理 1（架构选择律）
延续 v3。编译决策律（推论）：光滑度估计 ŝ 下码选择阈值——"谁值得花比特"。

### 定理 2（分层——King Theorem）
延续 v3。三层 + universal 线 + necessity（capacity 坐标形式）。层 1 的闭式传播机制显式引用 CROWN/DeepPoly 形态（不完备性谱 IBP<CROWN<LP<MILP 作为"验证成本光谱"的支持材料）。

### 定理 3（权衡——v4 类级限定）
> (i) **常数购买（类级限定）**：W^k 上，层 1（固定结构）容量常数 c_fix；结构最优（自由节点/数据相关网格）常数 c_* < c_fix。**类级最优**（对一切 f ∈ W^k 同时）需遍历节点集/结构 ⟹ V_arch 指数（显式归约：Thm 5 型 NP-hard / HLZ 型 PSPACE-hard）；**实例级**（给定节点集）误差界闭式可算（SOS 反例防御：承认实例级廉价，指数声明只在类级）。
> (ii) 类适配免费性（延续 v3）。
> (iii) 例外：量化部署位精确验证 PSPACE-hard（HLZ）⟹ V_B 指数 ⟹ 证书必须构建时生成、部署时只检查（PCC 分裂）。

---

## 3. 论证链（v4：Buss-Kapron 排除论证）

necessity 链 4 步延续 v3（表达式闭式性 → packing 逐实例注入 k 阶量 → M_k O(1) 可算 → 闭式传播）。

**Buss-Kapron 反例的适用性排除（v4 新增）**：Buss & Kapron（STOC 2002）证明存在类型二 functional，其证书大小与 sequential 查询成本**指数分离**（B-continuous 但非 B^n-sequential）。**不适用理由**：其对象是逐点 functional 的查询对话（查询-响应序对验证）；我们的对象是**类级 sup-norm 部署界**（β ≥ sup_x |f̂ − D(r)|，packing 线耦合）+ **闭式传播族**（规则③ 型加法结构）。两类设定不相交——必要性主张限定于后者（相对完备性，底线 2 延续）。

---

## 4. 归位映射（延续 v3，微调）
- T-I：激活级 + 插值/affine 类内 minimax + CP 规则 ②③ 来源
- Thm 5：定理 2 层 3 机制引理（**类级归约**——实例级误差闭式可算，类级最优指数）
- HLZ/GHL：定理 3 (iii) + V_B 维度
- CROWN/DeepPoly：层 1 闭式传播的已有形态（引用，非竞争）
- 三分法：定理 2 坐标化对象

---

## 5. 竞争者区分（v4 大表——"构件有先例、理论无先例"）

**逼近侧（完全让位）**：KT 1959 / Bronshtein / Zhu-Lafferty / OSB 2026 / Gonon 2023。

**验证侧构件（全部有先例，显式引用即免疫）**：

| 构件 | 先例 | 区分 |
|---|---|---|
| 证书演算（规则+绝对 soundness+机器可检） | Gappa / Sollya / LeanCert | 他们单函数实例界；我们类级+容量+necessity |
| 证书架构 | PCC / PCA / credible compilation / translation validation | 无容量维度；Rinard 2026 检查成本实测 = 验证成本经验测量先例 |
| 验证成本解耦 | SNARK/KLVW / GKR / PCP | 计算/统计 soundness + 电路对象；我们完美确定性 + 实数类级界 |
| 闭式传播 | CROWN / DeepPoly / IBP | 层 1 机制形态（引用）；无层化/容量 |
| 资源-保证单调 | Henzinger-Saraç（LICS 2021） | 在线监控属性精度 vs 离线类级证明长度 |
| 证书长度理论 | certificate complexity / Buss-Kapron | 布尔/逐点查询 vs 连续类级部署界（排除论证 §3） |
| 验证硬度 | HLZ 2021 / POPL 2025 / Sistla-Clarke 1985 | 层 3 机制 + 层 1 稀缺性支持 |
| SOS 精确有理证书 | Kaltofen-Li-Yang-Zhi | 实例级廉价（承认）；类级最优指数（不冲突） |

**验证侧容量/层化理论：无先例（唯一"无先例"声明点）**——红线②。

---

## 6. 三条红线（对抗复查定，论文写作纪律）

1. CP 章节以 Gappa/Sollya/LeanCert 为 related work 起点（不是脚注，是段落级起点）；
2. "无先例"只在"类级容量/层化理论"层使用（其余全部"构件有先例、理论无先例"句式）；
3. Thm 5 类级归约写全或降级为猜想（不允许直觉式指数声明）。

---

## 7. 状态与下一步

- **P1 完成**：三定理 v4 定稿（类级限定、CP 重定位、防御三件套）；necessity 链闭合 + Buss-Kapron 排除；对抗复查通过（漏洞全补）。
- **P2 数值验证（下一步）**：三件套 (i) packing 线常数匹配与排序 c_* < c_fix < c_k（合成 W^k 类 B 扫描）；(ii) 三层 (V_B, ε) 分离曲线（LUT/投影 = 层 1、自由节点 = 层 3 类级、ReLU = 层 2）；(iii) 决策律阈值扫描（ŝ → 码选择）。
- 严格证明书写 → P3。

## 8. 三个诚实底线（延续）

1. 训练网络非收缩（γ=[15.4,5.3]）；
2. 三分法 necessity 的 P≠NP 版 conjecture 保留（v4：necessity 限定"CP 内相对"+"类级部署界闭式传播族"——Buss-Kapron 设定外）；
3. 手写编号脆弱——setcounter 链清单。

## 9. 执行状态（P3 完成 2026-08-04）

- ✅ P3 论文重组完成：section_capacity.tex（Thm 17-19 + CP + related work 边界）插入 trichotomy 后；Abstract/Intro/Conclusion 加容量主线；references.bib +15 条；编译 **87 页 0e0w 0 undefined**；编号链 Thm 17/18/19 正确（50 labels）
- ✅ P2 三脚本（verify_capacity/stratification/decision_law）JSON 落盘 + 回归 9/9
- 🔴 待：REVIEWER-FAQ v4（容量防御）；最终 review-paper 全量；cover letter 更新；author bio/图清单
- 三条红线（§6）为 P3 写作纪律，已遵守（related work 段以 Gappa 为起点、无先例仅限类级容量层化层、Thm 5 类级归约表述）

*最后更新：2026-08-04 | 板板 + Claude | v4+P3：论文重组完成 87 页 0e0w，P4 投稿前准备*
