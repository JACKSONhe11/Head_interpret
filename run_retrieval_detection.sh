#!/bin/bash

# Retrieval Head Detection 运行脚本
# 使用方法: ./run_retrieval_detection.sh <model_path> <s_len> <e_len> [model_provider] [model_name_suffix]

# 检查参数
if [ $# -lt 3 ]; then
    echo "用法: $0 <model_path> <s_len> <e_len> [model_provider] [model_name_suffix]"
    echo ""
    echo "参数说明:"
    echo "  model_path: 模型路径或 HuggingFace 模型 ID（必需）"
    echo "  s_len: 起始上下文长度（必需）"
    echo "  e_len: 结束上下文长度（必需）"
    echo "  model_provider: 模型提供者，默认 'LLaMA'（可选）"
    echo "  model_name_suffix: 模型名称后缀（可选）"
    echo ""
    echo "示例:"
    echo "  $0 meta-llama/Llama-2-7b-chat-hf 1000 50000 LLaMA"
    echo "  $0 /path/to/model 1000 10000 LLaMA custom"
    exit 1
fi

MODEL_PATH=$1
S_LEN=$2
E_LEN=$3
MODEL_PROVIDER=${4:-"LLaMA"}
MODEL_NAME_SUFFIX=${5:-""}

# 检查必要目录
if [ ! -d "faiss_attn" ]; then
    echo "错误: 找不到 faiss_attn 目录"
    exit 1
fi

if [ ! -d "haystack_for_detect" ]; then
    echo "错误: 找不到 haystack_for_detect 目录"
    exit 1
fi

if [ ! -f "haystack_for_detect/needles.jsonl" ]; then
    echo "错误: 找不到 haystack_for_detect/needles.jsonl 文件"
    exit 1
fi

# 创建输出目录
mkdir -p results/graph
mkdir -p contexts
mkdir -p head_score

echo "=== Retrieval Head Detection ==="
echo "模型路径: $MODEL_PATH"
echo "上下文长度范围: $S_LEN - $E_LEN"
echo "模型提供者: $MODEL_PROVIDER"
if [ -n "$MODEL_NAME_SUFFIX" ]; then
    echo "模型名称后缀: $MODEL_NAME_SUFFIX"
fi
echo ""

# 构建命令
CMD="python retrieval_head_detection.py"
CMD="$CMD --model_path $MODEL_PATH"
CMD="$CMD --s_len $S_LEN"
CMD="$CMD --e_len $E_LEN"
CMD="$CMD --model_provider $MODEL_PROVIDER"

if [ -n "$MODEL_NAME_SUFFIX" ]; then
    CMD="$CMD --model_name_suffix $MODEL_NAME_SUFFIX"
fi

echo "运行命令:"
echo "$CMD"
echo ""

# 运行
$CMD

echo ""
echo "=== 运行完成 ==="
echo "结果保存在:"
echo "  - results/graph/<model_name>/"
echo "  - head_score/<model_name>.json"

