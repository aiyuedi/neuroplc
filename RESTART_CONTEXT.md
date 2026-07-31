# NeuroPLC 图表重启上下文

> 若在新 session 中接手本项目，先读此文件，再执行下方「重启第一步」。

## 1. 项目定位

- **论文**：NeuroPLC —— PLC 梯形图到 SCL 的神经网络编译器
- **投稿目标**：IEEE Transactions on Industrial Informatics（TII）
- **图表总数**：16 张（fig01–fig16 + fig1_overview + fig2_compiler_arch）
- **当前状态**：第一轮图表已生成，但质量未达顶刊标准，需**全部重做/深度优化**

## 2. 关键路径

| 用途 | 路径 |
|------|------|
| 论文根目录 | `D:\neuroplc-paper\` |
| LaTeX 主文件 | `D:\neuroplc-paper\paper\main.tex` |
| 图表最终输出 | `D:\neuroplc-paper\paper\figures\final\` |
| 图表源数据（CSV） | `D:\neuroplc-paper\paper\figures\source_data\` |
| MATLAB 生成脚本 | `D:\neuroplc-paper\code\figures\neuroplc_final.m` |
| MATLAB Python Builder | `D:\neuroplc-paper\code\figures\_build_matlab.py` |
| Origin 统计图脚本 | `D:\neuroplc-paper\code\figures\origin_figures.py` |

## 3. 图表清单与工具分配

| 编号 | 图题 | 推荐工具 | 备注 |
|------|------|----------|------|
| fig1 | 系统总览 | TikZ / draw.io → SVG | **当前太空，必须加内部模块细节** |
| fig2 | 编译器架构 | TikZ / draw.io → SVG | 同上，需体现数据流 |
| fig01 | C2BV 基函数 | MATLAB | 理论推导图 |
| fig02 | 正确性验证 | MATLAB | 组合电路真值表/波形 |
| fig03 | DA 紧性 | MATLAB | 双轴对比 |
| fig04 | Sharp Bound | MATLAB | 边界函数 |
| fig05 | DA vs IA 边界 | MATLAB | **当前左图 log 轴刻度有问题，右图 y 轴范围太宽** |
| fig06 | Adaptive LUT | MATLAB | 分段逼近示意 |
| fig07 | DA Scaling | MATLAB | 缩放误差分析 |
| fig08 | Segment Bounds | MATLAB | 分段边界 |
| fig09 | WCET 拆解 | MATLAB / Origin | 堆叠柱状图或瀑布图 |
| fig10 | 混淆矩阵（教师/学生）| MATLAB | **文字重叠严重，黑底白字需改** |
| fig11 | t-SNE 特征 | MATLAB | 散点图，需 colorblind-safe 调色盘 |
| fig12 | 交叉验证 | Origin / MATLAB | 箱线图或柱状图 |
| fig13 | 模型对比 | Origin / MATLAB | 多模型指标对比 |
| fig14 | 跨域测试 | MATLAB | 柱状图 |
| fig15 | 安全监控器 | MATLAB | 状态机/时序图 |
| fig16 | SCL 代码 | TikZ / listings | 代码高亮，排版为主 |

## 4. 上一轮已知缺陷（必须修复）

1. **fig1_overview / fig2_compiler_arch**：只有彩色空框，没有内部子模块、数据流箭头、接口标注，看起来像 PPT 模板。
2. **fig05_da_vs_ia**：左图 `log₁₀` 纵轴刻度标签被截断或格式不对；右图柱状图 y 轴 0–1 范围太空，没有体现数据差异；网格线太密（dotted grid）。
3. **fig10_confusion_matrices**：百分比与样本数垂直堆叠导致重叠；黑底白字对比度过高；colorbar 占宽过大；对角线黑块与文字融合。
4. **全局字体**：未统一为 Arial/Helvetica，字号层级混乱（标题、轴标签、刻度、图例没有明显区分）。
5. **全局颜色**：使用 MATLAB 默认橙蓝，未做 grayscale 验证；缺少 colorblind-safe 调色盘（Okabe-Ito / Wong 2011）。
6. **导出格式**：上一轮只导出了 PNG，需补全 **PDF/EPS 矢量导出 + 600 DPI**，且字体必须嵌入。
7. **尺寸**：未按 IEEE 单栏 8.5 cm / 双栏 17 cm 设置 Figure Size。
8. **Grid**：网格线应为极淡或完全去除，避免 chartjunk。

## 5. 顶刊标准（IEEE TII）

- **尺寸**：单栏 8.5 cm（3.35 in），双栏 17 cm（6.69 in），高度按黄金比例或内容自适应。
- **字体**：Arial 或 Helvetica，最小 8 pt，推荐层级：标题 12 pt / 轴标签 10 pt / 刻度 8 pt / 图例 8 pt。
- **颜色**：Okabe-Ito 调色盘（蓝 #0072B2、橙 #D55E00、绿 #009E73、红 #CC79A7、黄 #F0E442）；打印前用 `rgb2gray` 验证可区分性。
- **线型/标记**：B/W 兼容——同一张图里的多条曲线必须用「颜色 + 线型（实线/虚线/点划线）+ 标记（圆/方/三角）」三重编码。
- **导出**：`exportgraphics(gcf,'figXX.pdf','ContentType','vector','Resolution',600)`，再转 EPS。
- **Data-ink**：去掉所有非数据墨水——边框（box off）、厚重背景、多余网格、3D 效果。
- **矢量网格**：若必须加网格，用 `GridAlpha=0.15` + `GridLineStyle='-'`，避免 dotted 在 EPS 里变粗黑块。

## 6. 重启第一步（新 session 立刻执行）

```
1. 读取 D:\neuroplc-paper\paper\figures\source_data\ 下的所有 CSV，确认数据列名。
2. 读取 D:\neuroplc-paper\code\figures\_build_matlab.py 和 neuroplc_final.m，了解上一轮代码结构。
3. 按「顶刊标准」重建 _build_matlab.py 中的 Style 结构体：
   - 尺寸改为厘米制（8.5 / 17 cm）
   - 字体统一 Arial，字号层级 12/10/8 pt
   - 颜色改为 Okabe-Ito
   - GridAlpha 0.15 或去掉
4. 优先修复缺陷最明显的 3 张图：fig1_overview、fig05_da_vs_ia、fig10_confusion_matrices。
5. 每张图导出 PNG 预览 + PDF 矢量，放在 D:\neuroplc-paper\paper\figures\final\。
6. 用 rgb2gray 验证颜色在灰度下仍可区分。
7. 最后更新 main.tex 中的 \includegraphics 路径。
```

## 7. 环境信息

- MATLAB R2025b（已装）
- Origin 2025b（已装，COM 接口可用）
- Python：`D:\anaconda.install\envs\cq\python.exe`（可写 builder）
- LaTeX：xelatex（MiKTeX，`D:\miktex\`）
- 编译命令：`xelatex -shell-escape main.tex`（在 `D:\neuroplc-paper\paper\` 目录执行）

---

**最后更新**：2026-07-10
**状态**：待优化，图表需全部重做
