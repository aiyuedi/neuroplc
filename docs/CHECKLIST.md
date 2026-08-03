# NeuroPLC 投稿前修复追踪表（P0，2026-08-03 建立）

> 来源：PRE_SUBMISSION_REVIEW_2026-08-03.md（7-Agent 审稿）。状态：⬜ 待办 / 🔄 进行中 / ✅ 完成。
> 目标：P0 结束（08-16）时 CRITICAL 全绿（除需新定理的 4 项转入 P1）。

## A. CRITICAL（必须修）

| # | 项 | 出处 | 状态 | 备注 |
|---|----|------|------|------|
| C1 | Theorem B Galois 连接不健全（f(x)=x 反例；Coq lemma 不可证） | A3#3, A4#1 | ✅ | 2026-08-03：α 加一阶项、γ 改右伴随加权包络、α∘γ=id 降级为 tightness、Coq 同步；编译 0w |
| C2 | Lemma 6/Thm 1 界 vs 实测 14.17/≈17；Box-Continuation 未声明 | A3#1, A2#1, A4#2 | ⬜ | ℓ∞/ℓ₁ 标注 + 对账表 |
| C3 | 三分法 necessity 标注仅 1 处；"unique" 到处断言 | A3#2#5 | ⬜ | 全篇一致化 + C² 证据修正 |
| C4 | γ 双重断言（0.182 vs [15.4,5.3]） | A3#4, A2#4 | 🔄 | **调查裁决：0.182 是陈旧手写断言（无代码支撑，因子全无法重现）；[15.4,5.3] 实测正确**。修复批次已发（删 0.182 声称、L_B 实测 2.21、FT-Stability 重算、E55/E56 弱化、MATLAB 图 FIXME） |
| C5 | E52 vs E9/Thm1 数值冲突；reconciliation 表不全 | A2#2#3, A4#3 | ⬜ | DA 五值对账 |
| C6 | "Lemma 1.6" 硬编码×4 不存在 | A2#5, A4#1 | ⬜ | → \ref{lem:per_op}(vi) |
| C7 | E67 差分测试不可复现 | A7 FAIL | ✅ | 三层修复完成（broadcast/测试特征/SCL 外推语义），实测 100%/0.47，JSON 已提交 |
| C8 | 锐常数 folklore 叙事 + Thm 9(3) 无证明 | A6 Req#1, A4#16 | ⬜ | P1 定理 I 消化（T-I） |
| C9 | 三分法 necessity 证明缺失 | A6 Req#2 | ⬜ | P1 定理 III（T-III） |
| C10 | 无收缩训练演示 + γ 未对账 | A6 Req#3 | ⬜ | P1/P3（谱归一化训练） |
| C11 | 部署证书缺口（47% 出域） | A6 Req#4, A3 | ⬜ | 加宽 LUT 重验证 或 精确声明 |
| C12 | "verified compiler" 边界未定义 | A6 Req#5 | ⬜ | 明确形式验证 vs 差分测试边界 |
| C13 | SiLU″ 公式错误 + 三个衍生声称全错 | A4#10, A3#13 | ⬜ | σ(1−σ)[2+x(1−2σ)], sup=0.5 |
| C14 | WCET 三套指令时序（7× 差异） | A3#12, A4#14 | ⬜ | 统一时序表 |
| C15 | Fixed-Depth Tradeoff 自相矛盾 | A3#7, A4#5 | ⬜ | γ<1 时 gap/bias 缩小 |
| C16 | Thm 4 数值实例化错（0.86 vs 2.4） | A4#8 | ⬜ | 重算 |
| C17 | 分辨率匹配律过度声称（经典律未引；非 minimizer） | A6, A4#7 | ⬜ | P1 定理 II（T-II）消化 |

## B. MAJOR 关键项（必改）

| # | 项 | 状态 |
|---|----|------|
| M1 | 论文 E67 数字与实际不符（0.24） | ✅ 已改 0.47（main.tex×3 + FAQ×3 + JSON） |
| M2 | 引用缺失：CompCert/Tsybakov/MRW/Nye/Kratsios/Giacobbe/NeurIPS-CMI；Tankman | ⬜ |
| M3 | "Six Theorems" remark Prop 编号全错 | ⬜ |
| M4 | NC.1-3 非真环境（\ref 错解析） | ⬜ |
| M5 | 手写 Remark 2/3/4 与自动编号碰撞 | ⬜ |
| M6 | ~50 处硬编码 Theorem~N → \ref | ⬜ |
| M7 | thm:greedy/deep/hot_swap 从未 \ref；36 方程未引用 | ⬜ |
| M8 | 术语 γ/margin/M₂/h/ε/L/m/C/B 单义化 | ⬜ |
| M9 | 未定义符号 ε_fp/C(k)/‖W‖_prod/P(KAN)/safety factor | ⬜ |
| M10 | tab:summary：E58 缺行/E21-51 无说明/V7=V52 重复/E53 乱序 | ⬜ |
| M11 | 5 图从未引用（fig02/03/04/05/09）| ⬜ |
| M12 | "fixed-point"→"discretized(LUT)"×5；three/four-tier 统一 | ⬜ |
| M13 | "depth-uniform O(L)"→O(1)；"cubically"→quadratically | ⬜ |
| M14 | Fourier-Z3 矛盾；Wavelet-KAN citekey 错 + 小波类型矛盾 | ⬜ |
| M15 | IEC 通用保证降级（格式级≠操作语义）；删"entire ecosystem" | ⬜ |
| M16 | 非干扰证明补 WCET(P) 前提 | ⬜ |
| M17 | NC.3 溢界修正（[−23.9,25.1]）| ⬜ |
| M18 | RBF-KAN/MNIST 声称删除或改计划中；"physical hardware" 修正 | ⬜ |
| M19 | E56 "14.6×"→15.3×；"confirms depth-uniform" 修正 | ⬜ |
| M20 | MLP 对比复用已撤回 0.0019 | ⬜ |
| M21 | 15 个 "first/unique" 优先权断言 → 验证或降级 | ⬜ |
| M22 | 实验-修复时间线声明（哪些实验 pre-fix）| ⬜ |
| M23 | 32 个小数值不一致（A2 Mi 列表）| ⬜ |
| M24 | 论文 language 修复（A1: 2 CRITICAL + 12 MAJOR + 24 MINOR）| ⬜ |
| M25 | bib 清理（删 ~24 未引条目 + 占位）| ⬜ |
| M26 | iFA 跨厂商实验（编译 backend_iec ST + 仿真）| ⬜ |
| M27 | 4 个旧 SCL 文件清理 + IEC ST 乱码 | ⬜ |

## C. 状态汇总

- ✅ 完成：C7, M1
- 🔄 进行中：—
- 其余：⬜

*更新：2026-08-03 | 建立于 P0 首日*
