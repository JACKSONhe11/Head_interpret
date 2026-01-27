# Head Interpret

本项目提供了用于检测和分析 Transformer 模型中不同类型 attention heads 的工具。

## 环境配置

- Python 3.10.19
- CUDA 11.8 (如果使用 GPU)
- conda 或 miniconda

使用 conda 环境文件（推荐）

项目提供了完整的 conda 环境配置文件，可以快速重建环境：

```bash
# 使用简化版本（推荐，跨平台兼容性更好）
conda env create -f environment_rh_no_builds.yml

# 或使用完整版本（包含构建信息，适用于相同系统）
conda env create -f environment_rh.yml
```

创建环境后，激活环境：

```bash
conda activate rh
```



### 常见问题

1. **CUDA 版本不匹配**
   - 如果系统 CUDA 版本不同，需要安装对应版本的 PyTorch
   - 查看 [PyTorch 官网](https://pytorch.org/) 获取正确的安装命令

2. **依赖冲突**
   - 如果遇到依赖冲突，建议使用新的 conda 环境
   - 可以使用 `conda env export` 导出当前环境配置

3. **Flash Attention 相关问题**
   - 某些模型需要 Flash Attention，如果遇到问题，可以在代码中设置 `use_flash_attention_2=False`


- `faiss_attn/source/` - 自定义模型实现
- `data_cot/` - CoT 数据处理
- `head_score_all/` - 检测结果保存目录

## Head 类型选择

项目支持检测以下类型的 attention heads：

### 1. **retrieval_head** - 检索头
- **功能**: 用于长上下文中的信息检索
- **检测方法**: 通过 needle-in-haystack 测试检测
- **适用场景**: 分析模型如何在长文本中定位和检索特定信息

### 2. **previous_token_head** - 前一个 Token 头
- **功能**: 关注前一个 token
- **检测方法**: 通过 attention pattern 匹配检测
- **适用场景**: 分析模型的序列建模能力

### 3. **duplicate_token_head** / **duplicate_head** - 重复 Token 头
- **功能**: 关注相同 token 的重复出现
- **检测方法**: 通过 attention pattern 匹配检测
- **适用场景**: 分析模型如何识别和处理重复模式

### 4. **induction_head** - 归纳头
- **功能**: 关注重复模式后的下一个 token
- **检测方法**: 通过 attention pattern 匹配检测
- **适用场景**: 分析模型的模式识别和归纳能力

### 5. **iteration_head** - 迭代头
- **功能**: 用于链式思维推理中的迭代计算
- **检测方法**: 通过 CoT 数据集和 invariance 指标检测
- **适用场景**: 分析模型在推理任务中的计算模式

### 6. **truthfulness_head** - 真实性头
- **功能**: 能够区分真实答案和虚假答案
- **检测方法**: 通过 TruthfulQA 数据集和逻辑回归探针检测
- **适用场景**: 分析模型对真实性的判断能力
- **注意**: 需要先运行激活值生成脚本

### 7. **all** - 检测所有类型
- **功能**: 运行所有 head 类型的检测
- **返回**: 字典格式，键为 head 类型，值为对应的 heads 列表

## 模型选择

### Llama 系列模型

支持的 Llama 模型包括：

- `meta-llama/Llama-2-7b-hf` - Llama-2 7B 基础模型
- `meta-llama/Meta-Llama-3-8B-Instruct` - Llama-3 8B 指令微调模型
- 其他 Llama 系列模型

### Pythia 系列模型

支持使用 Pythia 模型和特定 checkpoint：

- **默认模型**: `EleutherAI/pythia-6.9b-deduped`
- **支持的 checkpoint**: `step3000`, `step10000`, `step143000` 等
- **使用方法**: 需要设置 `--use_pythia` 和 `--pythia_checkpoint` 参数

## 运行代码

### 基本用法

```bash
# 激活环境
conda activate rh

# 进入项目目录
cd Head_interpret

# 运行检测（使用默认参数）
python head_recog.py
```

### 运行示例

#### 1. 检测 Retrieval Head（Llama 模型）

```bash
python head_recog.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --head_type "retrieval_head" \
    --save_path "head_score_all"
```

#### 2. 检测 Iteration Head（Llama 模型）

```bash
python head_recog.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --head_type "iteration_head" \
    --save_path "head_score_all"
```

#### 3. 检测所有 Head 类型（Llama 模型）

```bash
python head_recog.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --head_type "all" \
    --save_path "head_score_all"
```

#### 4. 使用 Pythia 模型检测 Iteration Head

```bash
python head_recog.py \
    --use_pythia \
    --pythia_model_name "EleutherAI/pythia-6.9b-deduped" \
    --pythia_checkpoint "step3000" \
    --head_type "iteration_head" \
    --save_path "head_score_all"
```

#### 5. 检测 Pattern-based Heads（Previous Token, Duplicate, Induction）

```bash
# Previous Token Head
python head_recog.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --head_type "previous_token_head" \
    --save_path "head_score_all"

# Duplicate Token Head
python head_recog.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --head_type "duplicate_token_head" \
    --save_path "head_score_all"

# Induction Head
python head_recog.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --head_type "induction_head" \
    --save_path "head_score_all"
```

#### 6. 检测 Truthfulness Head

```bash
python head_recog.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --head_type "truthfulness_head" \
    --save_path "head_score_all" \
    --dataset_name "tqa_mc2" \
    --num_fold 2 \
    --num_heads 100
```

### 常用参数说明

#### 基本参数

- `--model_name`: 模型名称（默认: `meta-llama/Meta-Llama-3-8B-Instruct`）
- `--head_type`: Head 类型（默认: `retrieval_head`）
- `--save_path`: 结果保存路径（默认: `head_score_all`）
- `--rerun` / `--no-rerun`: 是否重新运行（默认: `--rerun`）

#### Retrieval Head 参数

- `--s_len`: 起始长度（默认: 1000）
- `--e_len`: 结束长度（默认: 5000）
- `--context_lengths_num_intervals`: 上下文长度间隔数（默认: 20）
- `--document_depth_percent_intervals`: 文档深度百分比间隔数（默认: 10）

#### Truthfulness Head 参数

- `--dataset_name`: 数据集名称（默认: `tqa_mc2`）
- `--num_fold`: 交叉验证折数（默认: 2）
- `--num_heads`: 选择的 top heads 数量（默认: 100）
- `--seed`: 随机种子（默认: 42）
- `--val_ratio`: 验证集比例（默认: 0.2）

#### Pythia 模型参数

- `--use_pythia`: 使用 Pythia 模型（需要与 `--pythia_checkpoint` 一起使用）
- `--pythia_model_name`: Pythia 模型名称（默认: `EleutherAI/pythia-6.9b-deduped`）
- `--pythia_checkpoint`: Pythia checkpoint revision（例如: `step3000`, `step10000`）

### 输出结果

检测结果会保存在 `save_path` 目录下，按模型名称组织：

```
head_score_all/
├── Llama-2-7b-hf/
│   ├── Llama-2-7b-hf_retrieval_head.json
│   ├── Llama-2-7b-hf_previous_token_head_custom_abs.pt
│   ├── Llama-2-7b-hf_duplicate_token_head_custom_abs.pt
│   ├── Llama-2-7b-hf_induction_head_custom_abs.pt
│   └── ...
└── pythia-6.9b-deduped/
    ├── pythia-6.9b-deduped_iteration_heads_inv_gt_0.70_sorted.npy
    └── ...
```

### 在 Python 代码中使用

```python
from head_recog import detect_heads

# 检测单个 head 类型
heads = detect_heads(
    model_name="meta-llama/Llama-2-7b-hf",
    head_type="retrieval_head",
    save_path="head_score_all"
)
# 返回: [(10, 5), (15, 10), (20, 15), ...]  # List of (layer_idx, head_idx)

# 检测所有 head 类型
all_results = detect_heads(
    model_name="meta-llama/Llama-2-7b-hf",
    head_type="all",
    save_path="head_score_all"
)
# 返回: {"retrieval_head": [...], "iteration_head": [...], ...}

```

## 可视化

可视化功能请参考 `viz_head_all/` 目录下的相关脚本。
