# ModelScope GPU Training Guide

> 账号: aiyuedi | 免费额度: 36h GPU (24GB VRAM, CUDA 12.8) | 永不过期

## 为什么用 ModelScope

| 方案 | VRAM | 训练时间 (估计) | 费用 |
|------|------|:---:|:---:|
| 本机 CPU | 0 (系统内存) | Teacher ~2h, Student ~4h | 0 |
| **ModelScope GPU** | **24GB** | **Teacher ~10min, Student ~20min** | **0** |
| 买云GPU | 16-24GB | 同上 | ¥20-50 |

ModelScope 免费 36h, 跑完全部实验绰绰有余。

## 使用步骤

### 1. 打包项目 (本地)

在项目根目录:

```bash
# 不包含数据 (数据库太大)
zip -r neuroplc-code.zip code/ paper/ results/scl_output/ -x "*/__pycache__/*" "*.pyc" ".pytest_cache/*"

# 数据单独传 (48 files, 142MB)
zip -r neuroplc-data.zip data/raw/12k_DE/
```

或者用 git push 到 Gitee/GitHub 再 clone。

### 2. 上传到 ModelScope

1. 浏览器打开 https://modelscope.cn/my/mynotebook
2. 登录 (手机号 181****2957)
3. 点击「创建实例」→ 选择 **GPU** 规格
4. 上传 zip 包 或 git clone
5. 解压: `unzip neuroplc-code.zip && unzip neuroplc-data.zip`

### 3. 安装依赖

在 ModelScope 终端:

```bash
pip install torch numpy scipy scikit-learn pandas matplotlib seaborn \
            pyyaml tqdm onnx onnxruntime scienceplots mlflow pingouin \
            statsmodels pytest
```

### 4. 运行训练

```bash
cd neuroplc-paper

# 快速测试 (5 epochs, 验证 pipeline 无 bug)
PYTHONIOENCODING=utf-8 python code/train_on_modelscope.py --all --test-mode

# 正式训练 (全量)
PYTHONIOENCODING=utf-8 python code/train_on_modelscope.py --all
```

### 5. 下载结果

训练完后:

```bash
# 打包结果
tar -czf neuroplc-results.tar.gz results/

# 下载到本地
# ModelScope 界面 → 文件管理器 → 右键 neuroplc-results.tar.gz → 下载
```

放到 `D:\neuroplc-paper\results\` 替换本地 results 目录。

### 6. 本地验证

```bash
# 跑测试确认结果完整性
PYTHONIOENCODING=utf-8 python -m pytest code/tests/ -q -s

# 生成论文图表
PYTHONIOENCODING=utf-8 python code/visualize.py --all

# 编译论文
cd paper && pdflatex main.tex
```

## 实验时间预估

| 实验 | 时间 (GPU) | 说明 |
|------|:---:|------|
| 预处理 | ~2min | CPU, 离散熵较慢 |
| Teacher 训练 (80 epochs) | ~10min | 1D-CNN ~50K params |
| Student KAN KD (100 epochs) | ~20min | ~300 params but KD loss 复杂 |
| E1-E7 评估 | ~3min | 推理快 |
| 可视化 | ~1min | 图表生成 |
| **总计** | **~35min** | 远低于 36h 免费额度 |

## 常见问题

**Q: 实例超时关闭怎么办？**
A: ModelScope GPU 实例空闲 30min 会休眠。训练中不会关闭。训练完及时下载结果。

**Q: 数据太大上传慢？**
A: 数据只需传一次。后面只传代码 (几MB)。

**Q: pip install 失败？**
A: 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 用清华镜像。
