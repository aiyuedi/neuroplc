# Pre-Submission Referee Report — NeuroPLC (2026-08-05)

**Paper**: A Type Theory of Certifiable Neural Architectures: Sharp LUT Bounds, Complexity Separation, and Compile-Aware Generalization, with an Industrial PLC Instantiation
**Target**: IEEE TNNLS (theory-first)
**Current state**: main.tex 5707 行，xelatex 编译 **91 页 / 0 errors / 16 overfull boxes**，无 undefined citation/reference（英文主文件）。
**审查对象**: main.tex 全章节 + section_svnn_theorems.tex（E-T9 双层证书）+ section_capacity.tex + docs/REVIEWER-FAQ.md + docs/cover_letter.md + docs/figure_list.md
**方法**: 全文通读 + 编译验证（aux 编号链）+ 12 组关键数字全位置扫描 + Claim-Evidence 审计 + 双层证书逻辑审查。

> 注意：审查期间 docs/ 下 FAQ 与 cover letter 正在被并发修改。首轮读取时发现的 FAQ Q27 标题（66x/763x）、FAQ Q29 与 cover letter 的 3L safety（4.4×）在复查时**已修复**（42.5x/493x、6.1×）。本报告行号以复查时刻（2026-08-05）的文件状态为准，提交前需重新核验一遍 docs。

---

## 一、执行摘要

双层证书体系（纯理论 tier 0.288/2.34× + 实测校准 tier 0.058/11.6×、0.026/26×）是本次升级的核心，其**宏观叙事成立**：0.288 的推导要素（M₂、per-edge Lipschitz、K）均可由模型参数设计时计算，声称"不依赖经验地板"有据（0.288 > 实测 0.053，5.4× headroom）；soft-contractive 的 γ 压缩、Box-Continuation 99.9% 覆盖、Z3 512/512、depth 扩展 6.1× 等均有命名脚本支撑。

但存在 **4 个阻塞级缺陷（P0）**，其中 2 个（tab:cross_domain、tab:scalability_grid）是 2026-08-04 审稿已列为 CRITICAL 且**至今未修复**的陈旧数字块；另外 2 个是证书体系自身的问题：**"0.058/0.026 实测地板证书"的推导公式全文缺失、无法复现，且其与 0.288 的关系、δ_fp32 ≤ 3×10⁻⁵ 与 fp32/fp64 两个 tier 之差（0.032）无法自洽**；以及 **da_bounds_summary 脚注"该 checkpoint 不存在设计时证书"与表格自身的 0.66/1.0× 证书行直接矛盾**。

**Preliminary Recommendation**: Revise（P0 全清前不要送审）。理论主体（19 定理、容量框架、sharp constants）经上一轮 6-agent 审计 + 本轮抽查，脚本-数字对账仍然干净；问题全部集中在**证书 tier 的表述纪律**与**修订一致性**。

---

## 二、六维评分（1–10）

| 维度 | 得分 | 依据 |
|------|------|------|
| 新颖性 | 9.0 | 类型系统框架、sharp 常数 c_k=Q_k/k!（c₃ 纠错 1/8）、ε-分离、容量分层、双层证书均为文献中未出现过的组合 |
| 严谨性 | 7.0 | 定理层面对账干净（E63–E68、E-T1–E-T11 脚本全部 PASS）；但证书 tier 的数字来源（0.058/0.026/0.0014/0.0078/0.039）无公式、方法论双轨（0.177 vs 0.218）并存导致 0.66/1.0× 与"无证书"并存 |
| 影响力 | 8.0 | KAN→PLC 端到端证书链 + 部署配方（contractive training）对工业安全社区有实质价值 |
| 清晰度 | 6.0 | 定理/图编号乱序（Thm 14 在第 20 页、Thm 1 在第 51 页；Fig.1=证书全景、Fig.7=流水线）；"sound/expected/floored" 术语在摘要、表、FAQ 中口径不一；tab:cross_domain 与全文矛盾 |
| 可复现性 | 8.5 | 每个定理命名脚本 + checkpoint 发布 + 3 命令复现；扣分点：0.058/0.026 公式未写入正文，读者无法从论文文本复现旗舰数字 |
| 完整性 | 7.5 | 19 定理/17 图/41 表/FAQ 20+9 问；扣分：E58 缺失于总表、CN 镜像严重陈旧（5 个 bib key 缺失→编译 [?]） |

---

## 三、必须修复（P0，阻塞投稿）

### P0-1 tab:cross_domain 整表陈旧，与全文自相矛盾 — main.tex:5166-5199
- **行 5187**：`SVNN Cond.~3 & Yes & Yes & Yes & Yes & Yes & Yes`。与 main.tex:3903（tab:xjtu_ft 脚注："SVNN Condition~3: **not satisfied** by this checkpoint, γ=[15.4,5.3]>1"）、与全部 γ>1 讨论（main.tex:2388-2397、section_svnn_theorems.tex:299、335）**直接矛盾**。上一轮审稿 CRITICAL#6 已列此条，未修复。
- **行 5186**：CWRU 列 `Safety margin 4.5×`。全文口径为 DA 1.0×/1.02×（main.tex:4008、4239）、full-φ 包络 0.3×–0.4×（main.tex:4033-4035）。4.5× 无任何出处（疑似旧 Δ=0.15 时代的残留）。同列 FourierKAN 2.9×、WaveletKAN 5.6× 与 E-T10/E-T11 的 4.9×/2.8×（main.tex:4019-4020、4425-4426）不一致；ChebyKAN 1.1× 与 fig02_verification 图注"Safety margins ≥ 2× for **all** C²-BV"（见 P1-4）冲突。
- **修复建议**：整列重算——CWRU 行改为"1.0× (DA,char) / 0.3× (full-φ)"或直接删除该列；Cond.3 行改为"Conditions 1–2 preserved; Cond.3 not satisfied (γ>1)"；MNIST 行补 E42 的 DA 0.1199；Fourier/Wavelet 行改为 4.9×/2.8×。

### P0-2 tab:scalability_grid 陈旧数字块 — main.tex:3587-3609
- **行 3601**：`G=15 ... DA Bound 0.0215 ... Safety 584×`。全文重算后的 N=15 DA 界为 **0.66**（main.tex:2002、4008），0.0215×L_net(162.2)=3.49≠0.66；safety 584× 与 0.675/0.66=1.02× 差 570×。G=4 行 safety 27× 与 0.675/0.4683=1.44 也不自洽。上一轮审稿 CRITICAL#6 已列，未修复。
- **修复建议**：用 `verify_da_bounds_recomputed.py` 常数重算整表（DA 界 ∝ 1/G² 但以 0.66@G=15 为锚），或删除该表并把"grid resolution 轴"改为引用 tab:blind_spot（该表已是重算口径）。

### P0-3 da_bounds_summary 脚注与表格行互相矛盾 — main.tex:3998-4069
- **行 4030-4035** 脚注："the full-φ float64 recalibration (2026-08-04) gives expected/worst envelopes 1.70/2.29 (safety 0.4×/0.3×)—**the design-time certificate does not exist for this checkpoint**"。
- 同一张表 **行 4008/4012** 却是 "Thm 1 DA (recomputed) 0.66 / 1.0×"、"E40 compositional 0.66 / 1.0×"；摘要与正文多处宣称 "DA bound certifies classification ... 1.0×"（main.tex:1307-1316、2020-2024、3982-3985；FAQ Q19）。审稿人必然抓到"你的表说没证书，你的摘要说有证书"。
- 根因：M₂ᶜʰᵃʳ 双口径（E11 0.177 vs full-φ 0.218）。0.218/0.177 再放大 DA 界 → 0.81 > 0.675，证书消失。
- **修复建议**：在一处（建议 tab:da_bounds_summary 脚注或 §DA 末尾）用一句话钉死口径："0.66/1.0× 为 per-logit-element、in-domain、M₂ᶜʰᵃʳ(0.177) 口径的高概率 DA 界；full-φ float64 重校准（M₂ᶜʰᵃʳ=0.218）下期望/最坏包络为 1.70/2.29，无证书"——并让摘要的"1.0× thin margin"与脚注的"无证书"同页可读、互不打架。

### P0-4 实测校准 tier（0.058/0.026）推导缺失，且被冠以 "sound" 之名 — section_svnn_theorems.tex:392-461
- **0.058/0.026 无法从论文文本复现**：脚注（main.tex:4061-4069）给出 float64 行"per-function no-cancellation propagation floored by float64 simulator maxAE 0.023; theoretical envelope 0.0078"——但 0.026 ≠ max(0.0078, 0.023)=0.023，+0.003 无出处；float32 行 0.058 与 0.053 之差 +0.005 同样无出处。0.058/0.026 之比 ≈ 0.053/0.023（实测地板之比），强烈暗示证书=地板×~1.1，但 ×1.1 从未被解释。
- **δ_fp32 ≤ 3×10⁻⁵ 与 fp32/fp64 tier 差 0.032 冲突**（section_svnn_theorems.tex:415-418）：若 fp32 附加项仅 3×10⁻⁵，为何 fp32 sound tier（0.058）比 fp64（0.026）大 0.032？两种读法（a"0.058 是计算界"→与 3e-5 矛盾；b"0.058 是地板+裕量"→对未见输入不是 bound）都不自洽。
- **FAQ 与论文对 0.058 的描述互相矛盾**：FAQ Q27（docs/REVIEWER-FAQ.md:419）称 sound tier 为"no-cancellation IA envelope at the **scipy-exact worst-function M₂**, floored by maxAE"；论文（section_svnn_theorems.tex:419-420）称其为"measured-floor bound for the float32 SCL backend"。若按 FAQ 读法（worst-function M₂ 包络），其值必须 ≥ 0.288（0.288 已用 per-edge L_{k,j} 与 M₂ᶜʰᵃʳ），但 0.058 < 0.288，算术上不可能——两种描述必有一错。
- **"sound" 标签过界**：摘要（main.tex:195-197）、结论（main.tex:5652-5656）、tab:cert_thresholds（main.tex:5492-5494）、cover letter 均以 "11.6× sound certificate" 为头条，而唯一对**未见输入**成立的界是 0.288（2.34×）。对 13,714 个输入的实测地板不是界。
- **修复建议**（任选其一，推荐 a）：
  - (a) 给每个 tier 写一行显式公式：理论 tier = 无消项传播(Σᵢεᵢ 直加 + L_{k,j} 放大)@M₂ᶠᵘˡˡ⁻ᵠ = 0.288；校准 tier 改为"empirically validated tier：Tier-4 模拟器 13,714 输入实测 maxAE 0.053（float32）/0.023（float64），实测裕量 11.6×/26×"，**把 "sound" 一词只留给 0.288/2.34×**，全文（摘要/结论/tab:cert_thresholds/FAQ/cover letter）同步改词；
  - (b) 若 0.058 确是计算界，给出完整推导（含 fp32 逐操作舍入如何累积到 0.032 量级——这必须推翻"≲3×10⁻⁵"）。
- 连带修正：FAQ Q27 的描述必须与论文选定的口径一致。

---

## 四、应该修复（P1）

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| P1-1 | 全文（aux 实证） | **定理编号完全乱序**：阅读顺序为 Thm 2(p7)→8(p9)→5(p15)→6(p16)→14(p20)→15(p21)→7(p23)→9(p24)→13(p25)→10(p26)→16(p27)→17-19(p28-31)→A-E→**Thm 3(p49)→Thm 1(p51)**→4(p53)→11(p59)。"Theorem 1"（编译器语义保持，论文核心定理）在第 51 页才出现；**Thm 12 不存在**（11 直接跳 13）。FAQ/cover letter 引用 13/14/15/16 与论文一致，但审稿人翻到第 20 页看 Thm 14、第 49 页才见 Thm 3 会直接判"草稿状态" | 投稿前做顺序重编号（上一轮 DEFERRED 项，建议本轮执行）；至少删除 setcounter 手法并验证所有 \ref |
| P1-2 | main.aux（fig:cert_panorama=1 p19, fig:capacity_plane=2 p31, fig:overview=7 p39） | **图编号乱序**：Fig.1=证书全景（E-T9 段，第 19 页），Fig.7=端到端流水线（第 39 页）；fig16_scl_code 是 Fig.13。figure_list.md v3 自列 #16/#17 与 LaTeX 实际编号（1/2）不符 | 图按阅读顺序重排（流水线→编译器→C²-BV→...→证书全景放 §E-T9）；figure_list.md 改为与 aux 实际编号一致 |
| P1-3 | section_characterization.tex:128-150（fig04_sharp_bound 图注，编译后见 aux） | 图注自认陈旧："the dashed blue line is the illustrative contractive case γ=0.182 used in earlier versions---**stale**"。正文修好了，图还在给审稿人看道歉 | 重绘 fig04（删 γ=0.182 虚线），图注不再含 "stale" |
| P1-4 | fig02_verification 图注（aux fig:c2bv_verification） | "Safety margins ≥ 2× confirm deployability for **all** C²-BV architectures" vs tab:cross_domain ChebyKAN 1.1×（main.tex:5186） | 统一 safety margin 定义，或把 ≥2× 改为按架构列举 |
| P1-5 | main.tex:4015、5494；section_svnn_theorems.tex:402-403 | **0.0014 与 493× 算术不符**：0.675/0.0014 = 482 ≠ 493。要使 493× 成立需 Δ=0.00137 或 margin=1.38（≠脚注所称全表共享 1.35，main.tex:4026"same minimum inter-class margin (1.35, E6)"）。三数不可能同时成立 | 给出 soft-contractive checkpoint 自己的最小 margin；统一 0.0014/493×/margin 三数 |
| P1-6 | section_svnn_theorems.tex:448 | "2,188-line SCL (40.3 KB, **53.7%** of the S7-1200 work-memory budget)"：40.3/50 = 80.6%，53.7% 无出处（40.3/75 或 26.9/50 均不成立；TIA 实测为 45.2 KB/90.4%，main.tex:2708-2710） | 改为 80.6%（静态估计口径）或引用 TIA 实测 45.2 KB/90.4% |
| P1-7 | main.tex:4035、section_svnn_theorems.tex:429-430 | 释放 checkpoint 的 "measured maxAE (0.52)" 与 E6 同 checkpoint 的 MaxAE **3.65**（main.tex:3520）并存且 0.52 无测量口径（测试集？in-domain？）。2.29/0.52=4.4 只在 in-domain 成立 | 加 in-domain 限定并注明 0.52 的测量协议（参照 main.tex:2206-2211 的 Remark 口径） |
| P1-8 | main.tex:4770 | "per-layer amplification envelope γ² = 28.1"：γ=[15.4,5.3] 下 28.1=5.3²，而全文其他处取 γ=15.4（如 section_svnn_theorems.tex:300 "γ^{L-1}=15.4"）；一致口径应为 15.4²=237（结论"5.66 亚指数"依然成立，但数字错了） | 改为 15.4²=237 或明示取 5.3 的理由 |
| P1-9 | main_cn.tex:1357、1550、1773；section_svnn_cn.tex:24、145、220 | **CN 镜像严重陈旧**：(a) DA 0.411→0.165 写成 0.064→0.049（1357）；(b) DA 表 0.1199/0.1196/0.0493/0.1242（1773）为旧值；(c) 53.4 KB（应为 45.2）；(d) 5 个 cite key（lutkan2025、adaptpolykan2025、tankman2026lipschitz、torchlean2025、markov1916limit）不在 references.bib → CN 版编译会出现 [?]；(e) section_svnn_cn.tex:145 仍含陈旧 γ=0.182/L_B=0.65 | CN 镜像与英文主文件全量同步，或明确标注"非投稿件" |
| P1-10 | 全文 | **91 页 single-column vs TNNLS 常规 14 页双栏**（≈28 页单栏）：上一轮即列为 submission-format blocker，本轮无变化 | 决定投稿策略（长文/增刊拆分/压缩），投稿前必须定案 |
| P1-11 | docs/REVIEWER-FAQ.md:419 | FAQ Q27 对 sound tier 的描述与论文矛盾（见 P0-4） | 与论文选定口径同步 |
| P1-12 | section_svnn_theorems.tex:359-361 | "11×/3.8× reduction from [15.4, 5.3]"：成分配对含糊。逐分量应为 15.4/1.03≈15×、5.3/1.38≈3.8×；"11×"=15.4/1.4 是跨分量相除 | 写明配对关系 |
| P1-13 | section_svnn_theorems.tex:397-398、435 | 释放 checkpoint 的 "no-cancellation amplification constant K=340" 与 Thm 1 的 L_net^IA=171.1（main.tex:2001）同为"无消项"常数却差 2 倍，未对账 | 定义 K 或对账 L_net^IA/DA=171.1/162.2 |
| P1-14 | section_svnn_theorems.tex:404-406 | "worst-function envelope (full-φ M₂ᵐᵃˣ, float64) = 0.039 (17.5×)"：0.039 < 0.058（float32 sound）< 0.288（理论 tier）——"最坏函数包络"小于"理论包络"，且 0.039/0.0078 均无对应表行，读者无法验证单调结构 | 归入 P0-4 的公式统一方案，或单独成行进 da_bounds_summary |

---

## 五、建议（P2）

1. **1.0× vs 1.02×**：tab:blind_spot N=15 行（main.tex:4239）为 1.02×（0.675/0.66=1.023），其余全部 1.0×（4008、1315、2021...）。统一舍入约定。
2. **2,187 vs 2,188 行**：main.tex:5345 "2,187 lines" vs 3874/3902/5228 及 E-T9 段的 2,188。核对 V2 输出实际行数。
3. **tab:cert_thresholds 末行**（main.tex:5495-5496）：Safety 列 1.4×（=0.675/0.4697）与 Evidence 列 "1.5× safety margin" 打架。
4. **E58 缺席总表**：main.tex:2025、2308、4895 引用 E58，但 tab:summary（4368-4442）无 E58 行（E58=per-function M₂ 分布/Z3 验证）。补行或删引用。
5. **FAQ Q10**（docs/REVIEWER-FAQ.md:191）：实验清单仍写 "E-T1–E-T3"（现为 E-T1–E-T11）。
6. **cover letter**（docs/cover_letter.md:21）："the **three** new capacity experiments (E-T4/E-T5)"——写了两个。
7. **fig_cert_panorama 图注**（section_svnn_theorems.tex:479）："median 0.218 → 0.051/0.052"，0.052 仅此一处（3L 的 M₂ᶜʰᵃʳ 正文未给出），补 3L 数据出处。
8. **16 个 overfull hbox**：最大两个——tab:xjtu_ft（main.tex:3890-3908，189pt）、tab:cert_thresholds（5479-5506，54.6pt）、main.tex:962-967（111pt）。修表宽。
9. **E-T9 段的 SCL 证据链**：soft-contractive checkpoint "compiles to 2,188-line SCL"（section_svnn_theorems.tex:447-448）未像 E55 那样声明 TIA 0e0w 实测；若 E-T9 SCL 未过 TIA 编译，请注明（审稿人会问）。
10. **"0.03/0.26 pp" 基线不一**（section_svnn_theorems.tex:384-386）：Fourier 用 99.99 作基线（0.03pp），Wavelet 用 99.93（0.26pp）；但 E60/E61 声称两者 CWRU 均为 100%（main.tex:527-532），按 100% 基线应为 0.04/0.33pp。注明基线选择。
11. **fig04 图注的 γ=0.182 道歉句**（P1-3 附带）：同图右侧 R(d)=√d/γ 仍用旧 γ 计算，重绘时一并处理。
12. **FAO Q7 的 "12.4"**（旧版）已改为 2.29 ✓（复查确认），无需处理——列入以防回归。

---

## 六、审稿人攻击面预测（Top 3 + 防御位置）

### 攻击 1（最致命）：「11.6× sound certificate」是实测地板，不是对未见输入的界
审稿人会从摘要（main.tex:195-197 "an 11.6× sound (no-cancellation, measured-floor) / 493× expected certificate"）与结论（5652-5656）读到一个"sound certificate 11.6×"，然后在 E-T9 段发现 0.058 的推导缺失、且 0.058 与 0.026 之差无法用其自称的 δ_fp32 ≤ 3×10⁻⁵ 解释 → 结论：**论文唯一的输入无关证书是 0.288/2.34×，旗舰数字被高估 5 倍**。
- **防御位置**：section_svnn_theorems.tex:406-422（理论 tier 段）——按 P0-4(a) 改造后，此处明确写"worst-case certificate = 0.288 (2.34×)；11.6×/26× 为 13,714 输入实测验证裕量"；FAQ Q27 同步。

### 攻击 2：「你自己的表格说释放 checkpoint 没有证书，摘要却说有 1.0× 证书」
da_bounds_summary 脚注（main.tex:4033）"the design-time certificate does not exist" vs 行 4008 "0.66, 1.0×" vs 摘要/FAQ 的 "thin 1.0× margin"。
- **防御位置**：按 P0-3 修复后在 §DA（main.tex:1318-1366）与 tab 脚注各放一句口径钉死句；FAQ Q19 已表态"marginal"，需与 new methodology 对齐。

### 攻击 3：「修订一致性」——陈旧表 + 乱序编号 = 草稿状态（desk-reject 风险）
tab:cross_domain "Cond 3: Yes"（P0-1）、tab:scalability_grid 584×（P0-2）、Fig.1=证书全景而 Fig.7=流水线（P1-2）、Thm 14 先于 Thm 1 出现且 Thm 12 不存在（P1-1）、fig04 图注自认 stale（P1-3）。审稿人 5 分钟内即可在正文里找到 3 处数字矛盾，无需读理论。
- **防御位置**：无——这是纯工程问题，投稿前用本报告的核对表逐条清零。

### 附带攻击：493× 算术（0.675/0.0014=482）、0.03/0.26pp 基线、E58 缺表、2,187/2,188——均为"审稿人用计算器 30 秒内可抓"的小数，建议全部清零。

---

## 七、数字一致性核对表

核对时间 2026-08-05；「判定」= 一致 ✓ / 不一致 ✗ / 存疑 ⚠。

| 数字 | 位置（文件:行） | 当前值 | 期望值 | 判定 |
|------|----------------|--------|--------|------|
| 0.288 / 2.34× | main.tex:4016；svnn_theorems:411-412；cover_letter:23 | 0.288, 2.34× | 0.675/0.288=2.34 ✓ 全文一致 | ✓ |
| 11.6× | main.tex:196,1315,4018,5492,5653；svnn_theorems:420,446,472；FAQ:419；cover_letter:23 | 0.058, 11.6× | 0.675/0.058=11.6；位置间一致 | ✓（但见 P0-4：口径） |
| 26× | main.tex:4017；svnn_theorems:421,472；cover_letter:23 | 0.026, 26× | 0.675/0.026=25.96 ✓ | ✓（推导缺失见 P0-4） |
| 6.1× (3L) | main.tex:197,4402,4778,5655；svnn_theorems:455；FAQ:429；cover_letter:23 | 0.110, 6.1× | 0.675/0.110=6.14 ✓；FAQ/cover letter 已从 4.4× 修正 | ✓ |
| 4.9× (Fourier) | main.tex:4019,4425；svnn_theorems:378；cover_letter:23 | 0.138, 4.9× | 0.675/0.138=4.89 ✓ | ✓ 文本；**✗ tab:cross_domain:5186 写 2.9×** |
| 2.8× (Wavelet) | main.tex:4020,4426；svnn_theorems:380；cover_letter:23 | 0.237, 2.8× | 0.675/0.237=2.85 ✓ | ✓ 文本；**✗ tab:cross_domain:5186 写 5.6×** |
| 2.29 / 0.3× | main.tex:4033,4068；svnn_theorems:428,469；FAQ:419 | 2.29, 0.3× | 0.675/2.29=0.295 ✓ 全位置一致 | ✓ |
| 1.70 / 0.4× | main.tex:4033,4069；svnn_theorems:469 | 1.70, 0.4× | 0.675/1.70=0.397 ✓ | ✓ |
| 0.0527 | 全库扫描（tex+docs） | 不存在 | — | ⚠ 无此数字；确认无需引入（近似值应为 0.053） |
| 0.675 | main.tex ×10（1305,2023,4239...）；svnn_theorems:468 | 0.675 | 半余量 m/2=1.35/2 一致 | ✓ |
| 3×10⁻⁵ | svnn_theorems:415-418；cover_letter:23 | δ_fp32 ≲3e-5 | 与 fp32/fp64 tier 差 0.032 冲突（P0-4） | ✗ 需补推导或改口径 |
| 493× | main.tex:196,1315,4015,5494；svnn_theorems:403,446；FAQ:416,419；cover_letter:23 | 0.0014, 493× | 0.675/0.0014=482≠493（P1-5） | ✗ |
| 0.058 / 0.053 | main.tex:198,4018,4066-4067,5653-5654；svnn_theorems:413,420；FAQ:419 | 0.058 / 0.053 (13,714 inputs) | 0.058−0.053=0.005 无出处（P0-4） | ⚠ |
| 0.026 / 0.023 | main.tex:4017,4063-4065；svnn_theorems:421-422 | 0.026 / 0.023 | 0.026≠max(0.0078,0.023)（P0-4） | ✗ |
| 0.0078 | main.tex:4065；svnn_theorems:471 | "theoretical envelope"（脚注）/ "per-function"（图注） | 两种标签并存，口径未定义 | ⚠ |
| 0.039 / 17.5× | svnn_theorems:404-405 | 0.039, 17.5× | 0.675/0.039=17.3 ✓ 但无表行、单调结构存疑（P1-14） | ⚠ |
| 0.110 / 0.019 / 36× | main.tex:4778-4779；svnn_theorems:454-455；FAQ:429 | 0.110(6.1×), 0.019(36×) | 0.675/0.110=6.14、0.675/0.019=35.5 ✓ | ✓ |
| 0.52 (释放 checkpoint maxAE) | main.tex:4035；svnn_theorems:429-430 | 0.52, "4.4× below envelope" | 2.29/0.52=4.4 ✓ 但 vs E6 MaxAE 3.65 缺口径（P1-7） | ⚠ |
| 1.0× / 1.02× | main.tex:4008,4012,1315,2021 vs 4239,4251,4267 | 1.0× / 1.02× 并存 | 0.675/0.66=1.023 | ⚠ 统一舍入 |
| 5.9 (M₂ᵐᵃˣ DA) | main.tex:2300-2303,4009；FAQ:419 | 5.9, <1 | 0.675/5.9=0.11 ✓ 一致 | ✓ |
| γ=[1.03,1.38] | svnn_theorems:359；main.tex:4424,4058-4059；FAQ:419 | 98.5%, 1.4pp | 一致 | ✓（11×/3.8× 配对见 P1-12） |
| γ=[0.99,1.03,1.02] (3L) | svnn_theorems:453；main.tex:4777；FAQ:429 | 98.6% | 一致 | ✓ |
| 99.9% Box coverage / 0.1% | main.tex:196,5654,5493；svnn_theorems:438-441；FAQ:424 | 99.9% / 47.1%→0.1% | 一致 | ✓ |
| Fourier 97.3% / Wavelet 100% coverage | svnn_theorems:379,381 | 97.3% / 100% | 仅此一处，文本内一致 | ✓ |
| 98.5% / 99.96% / 99.67% | main.tex:4424-4426；svnn_theorems:358,377,380；cover_letter:23 | 一致 | 一致 | ✓（0.03/0.26pp 基线见 P2-10） |
| 40.3 KB / 53.7% | svnn_theorems:448 | 40.3 KB, 53.7% | 40.3/50=80.6%（P1-6） | ✗ |
| 3.86 ms / 25.9× / 3.9% | main.tex:3032-3034,515,347；svnn_theorems:449-450；FAQ:365,324 | 一致 | 3,855.2μs ✓ | ✓ |
| 2,188 vs 2,187 行 | main.tex:5345 vs 3874,3902,5228 | 2,187 / 2,188 | 核对 V2 输出 | ⚠ P2 |
| 584× (G=15) | main.tex:3601 | 0.0215, 584× | 0.66, 1.02×（P0-2） | ✗ |
| 4.5× / 1.1× / 2.9× / 5.6× (cross_domain) | main.tex:5186 | 陈旧 safety margin | 1.0×/—/4.9×/2.8×（P0-1） | ✗ |
| Cond.3 "Yes" (cross_domain) | main.tex:5187 | Yes×6 | Not satisfied（γ>1）（P0-1） | ✗ |
| 28.1 (γ²) | main.tex:4770 | 28.1 | 15.4²=237（P1-8） | ✗ |
| K=340 vs 171.1 | svnn_theorems:397-398 vs main.tex:2001 | 340 / 171.1 | 需对账（P1-13） | ⚠ |
| E66 斜率 | main.tex:4411；svnn_theorems 无；FAQ:375 | −2.04/−2.06/−1.77/−1.03 | 一致 | ✓（FAQ 将 B-spline 配 −2.04、论文配 tanh，次要） |
| 5.66 (E56 DA) | main.tex:4402,4768；FAQ:429 | 5.66 | 一致 | ✓ |
| 2.75×/3.32×/2.10× | section_lut_sharp.tex:177,190-193；main.tex:4416；FAQ:241,348 | 一致 | 一致 | ✓ |

---

## 八、Claim-Evidence 审计摘要

| 强 claim | 证据链 | 判定 |
|---------|--------|------|
| "纯理论 tier 无经验系数"（0.288） | 要素（M₂、L_{k,j}、K）均可由模型参数设计时计算；0.288 > 实测 0.053（5.4× headroom）故不依赖地板 | ✓ 成立，但"first-principles"宜改为"model-parameter-derived"，且 0.288 的传播公式需显式写出（P0-4） |
| "soft-contractive 全链证书：Z3 512/512、99.9% 覆盖、2,188 行 SCL、WCET 3.86ms 不变" | verify_contractive_bounds.py、verify_box_coverage.py、编译链 2026-08-04 | ✓ 脚本齐备；SCL 是否过 TIA 0e0w 未声明（P2-9） |
| "19 定理全部有验证脚本" | E63–E68 + E-T1–E-T11 命名脚本 + results JSON | ✓（上一轮 6-agent 已验证脚本-数字精确对账） |
| "释放 checkpoint 有 1.0× DA 证书" | 0.66 < 0.675（M₂ᶜʰᵃʳ=0.177） | ⚠ 与 full-φ 口径（1.70/2.29 无证书）并存未调和（P0-3） |
| "11.6× sound（no-cancellation, measured-floor）" | maxAE 0.053/13,714 inputs | ✗ "sound"标签过界；0.058 推导缺失（P0-4） |
| "0.03/0.26 pp 有界幅值基证书几乎免费" | Fourier 99.96%、Wavelet 99.67% | ⚠ 基线选择不一（99.99 vs 99.93 vs E60/E61 的 100%）（P2-10） |
| "δ_fp32 ≤ 3×10⁻⁵ 闭环" | 0.5 ulp/操作逐操作传播 | ⚠ 无推导、与 0.058−0.026=0.032 冲突（P0-4） |

---

## 九、结论

理论层（sharp constants、ε-分离、容量分层、软收缩模型）经本轮与上轮双重审计仍站得住，脚本对账干净；阻塞项全部集中在**证书体系的表述纪律**（P0-3、P0-4）、**陈旧数字块**（P0-1、P0-2）与**编号秩序**（P1-1、P1-2）。P0 清零 + P1 主要项（1-5、7、8）处理 + docs 与主文件做一次最终同步后，可以送审。建议在修复后重新编译并做一次 grep 级数字终检（本报告核对表可作为 checklist）。

*Report generated 2026-08-05. 审查人：paper-reviewer（全量单agent深度审稿：通读 5707 行主文件 + 9 个章节文件 + 3 份投稿文档 + xelatex 编译验证 + 44 项数字对账）。*
