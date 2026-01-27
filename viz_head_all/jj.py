"""
Head Visualization Module

Provides a unified interface to visualize head detection results.
Simply provide head_type and model_name to generate visualizations.

地址设置说明：
=============
有两种方式设置地址：

方式1: 在文件开头配置（推荐，第44-75行）
------------------------------------
在文件开头的"地址配置区域"为每个 head 类型设置独立地址：
- previous_token_head_address
- duplicate_token_head_address  
- induction_head_address
- retrieval_head_address
- iteration_head_address
- truthfulness_head_address

如果设置为 None，将使用默认路径自动查找。
如果设置了具体路径，将优先使用该路径。

方式2: 调用函数时设置
-------------------
调用 visualize_heads() 函数时，可以通过以下参数设置地址：

1. base_dir (可选): 
   - 设置 head score 文件的输入目录（从哪里读取文件）
   - 如果不设置，默认使用: Retrieval_Head/head_score_all/{model_version}/
   - 示例: base_dir=Path("/path/to/head_scores/Meta-Llama-3-8B-Instruct")

2. output_path (可选):
   - 设置可视化结果的保存路径（保存到哪里）
   - 如果不设置，默认保存到: {base_dir}/visualization/{filename}.png
   - 示例: output_path=Path("/path/to/output/visualization.png")

使用示例：
---------
from pathlib import Path
from visualization import visualize_heads

# 使用默认地址（或文件开头配置的地址）
visualize_heads("all", "meta-llama/Meta-Llama-3-8B-Instruct")

# 自定义输入目录
visualize_heads("all", "meta-llama/Meta-Llama-3-8B-Instruct",
                base_dir=Path("/custom/input/path"))

# 自定义输出路径
visualize_heads("iteration_head", "meta-llama/Meta-Llama-3-8B-Instruct",
                output_path=Path("/custom/output/path.png"))
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Union

# ============================================================================
# 地址配置区域 - 请在这里填写每个 head 类型的文件路径
# ============================================================================
# 说明：
# - 如果某个地址设置为 None，将使用默认路径自动查找
# - 如果设置了具体路径，将优先使用该路径
# - 路径可以是相对路径或绝对路径
# ============================================================================

# Previous Token Head 文件地址
# 支持字符串或 Path 对象，如果为 None 则使用默认路径
previous_token_head_address: Optional[Union[Path, str]] = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_previous_token_head_custom_abs.pt"
# 示例: previous_token_head_address = Path("/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_previous_token_head_custom_abs.pt")
# 或者: previous_token_head_address = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_previous_token_head_custom_abs.pt"

# Duplicate Token Head 文件地址
duplicate_token_head_address: Optional[Union[Path, str]] = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_duplicate_token_head_custom_abs.pt"
# 示例: duplicate_token_head_address = Path("/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_duplicate_token_head_custom_abs.pt")
# 或者: duplicate_token_head_address = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_duplicate_token_head_custom_abs.pt"

# Induction Head 文件地址
induction_head_address: Optional[Union[Path, str]] = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_induction_head_custom_abs.pt"
# 示例: induction_head_address = Path("/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_induction_head_custom_abs.pt")
# 或者: induction_head_address = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_induction_head_custom_abs.pt"

# Retrieval Head 文件地址
retrieval_head_address: Optional[Union[Path, str]] = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_retrieval_head.json"
# 示例: retrieval_head_address = Path("/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_retrieval_head.json")
# 或者: retrieval_head_address = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_retrieval_head.json"

# Iteration Head 文件地址
iteration_head_address: Optional[Union[Path, str]] = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_iteration_heads_inv_gt_0.70_sorted.npy"
# 示例: iteration_head_address = Path("/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_iteration_heads_inv_gt_0.70_sorted.npy")
# 或者: iteration_head_address = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_iteration_heads_inv_gt_0.70_sorted.npy"

# Truthfulness Head 文件地址
truthfulness_head_address: Optional[Union[Path, str]] = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/truthfulness_head/truthfulness_head_Meta-Llama-3-8B-Instruct_avg.npy"
# 示例: truthfulness_head_address = Path("/home/jhe/Head_analysis/Retrieval_Head/head_score_all/truthfulness_head/truthfulness_head_Meta-Llama-3-8B-Instruct_avg.npy")
# 或者: truthfulness_head_address = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/truthfulness_head/truthfulness_head_Meta-Llama-3-8B-Instruct_avg.npy"

# ============================================================================

# Try to import plotly for visualization
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Install with: pip install plotly kaleido")
    
# Try to import neel_plotly for visualization (optional)
try:
    from neel_plotly import imshow
    HAS_NEEL_PLOTLY = True
except ImportError:
    HAS_NEEL_PLOTLY = False

# Fallback to matplotlib if available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


# Import visualization functions from the original file
import sys
import importlib.util

current_dir = Path(__file__).parent
original_viz_file = current_dir / "visualize_head_all_in_one_vertical_split_custom.py"

# Import functions from the original visualization file
if original_viz_file.exists():
    spec = importlib.util.spec_from_file_location("viz_module", original_viz_file)
    viz_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz_module)
    
    # Import all necessary functions
    load_head_scores = viz_module.load_head_scores
    load_truthfulness_heads = viz_module.load_truthfulness_heads
    load_iteration_heads = viz_module.load_iteration_heads
    get_top_k_heads = viz_module.get_top_k_heads
    create_highlighted_matrix = viz_module.create_highlighted_matrix
    visualize_with_neel_plotly = viz_module.visualize_with_neel_plotly
    visualize_with_matplotlib = viz_module.visualize_with_matplotlib
    decode_mask_value = viz_module.decode_mask_value
    encode_mask_value = viz_module.encode_mask_value
    create_combined_head_matrix = viz_module.create_combined_head_matrix
    visualize_combined_heads = viz_module.visualize_combined_heads
    visualize_combined_with_matplotlib = viz_module.visualize_combined_with_matplotlib
    visualize_combined_with_plotly = viz_module.visualize_combined_with_plotly
    visualize_head_scores = viz_module.visualize_head_scores
else:
    raise ImportError(f"Could not find visualization module: {original_viz_file}")


def get_head_score_paths(model_name: str, base_dir: Optional[Path] = None) -> dict:
    """
    Get file paths for all head score files based on model name.
    
    优先使用文件开头配置的地址（previous_token_head_address 等），
    如果地址为 None，则使用默认路径自动查找。
    
    Args:
        model_name: Model name, e.g., "meta-llama/Meta-Llama-3-8B-Instruct"
        base_dir: Base directory for head scores. If None, uses default location.
        
    Returns:
        Dictionary with file paths for each head type:
        {
            'previous_token_head': Path,
            'duplicate_token_head': Path,
            'induction_head': Path,
            'retrieval_head': Path,
            'iteration_head': Path,
            'truthfulness_head': Path
        }
    """
    model_version = model_name.split("/")[-1]
    
    # Default base directory
    if base_dir is None:
        # Try to find head_score_all directory relative to current file
        current_file = Path(__file__)
        # Go up to Retrieval_Head directory
        retrieval_head_dir = current_file.parent.parent
        base_dir = retrieval_head_dir / "head_score_all" / model_version
    
    paths = {}
    
    # Helper function to convert string to Path if needed
    def to_path(value):
        if value is None:
            return None
        if isinstance(value, str):
            return Path(value)
        return value
    
    # Previous Token Head - 优先使用配置的地址
    if previous_token_head_address is not None:
        paths['previous_token_head'] = to_path(previous_token_head_address)
    else:
        paths['previous_token_head'] = base_dir / f"{model_version}_previous_token_head_custom_abs.pt"
    
    # Duplicate Token Head - 优先使用配置的地址
    if duplicate_token_head_address is not None:
        paths['duplicate_token_head'] = to_path(duplicate_token_head_address)
    else:
        paths['duplicate_token_head'] = base_dir / f"{model_version}_duplicate_token_head_custom_abs.pt"
    
    # Induction Head - 优先使用配置的地址
    if induction_head_address is not None:
        paths['induction_head'] = to_path(induction_head_address)
    else:
        paths['induction_head'] = base_dir / f"{model_version}_induction_head_custom_abs.pt"
    
    # Retrieval Head - 优先使用配置的地址
    if retrieval_head_address is not None:
        paths['retrieval_head'] = to_path(retrieval_head_address)
    else:
        paths['retrieval_head'] = base_dir / f"{model_version}_retrieval_head.json"
    
    # Iteration Head - 优先使用配置的地址
    if iteration_head_address is not None:
        paths['iteration_head'] = to_path(iteration_head_address)
    else:
        # Search for iteration head files (may have different thresholds)
        iteration_pattern = f"{model_version}_iteration_heads_inv_gt_*.npy"
        iteration_files = list(base_dir.glob(iteration_pattern))
        if iteration_files:
            # Use the first one found (or could use the one with highest threshold)
            paths['iteration_head'] = sorted(iteration_files)[-1]  # Use highest threshold
        else:
            paths['iteration_head'] = None
    
    # Truthfulness Head - 优先使用配置的地址
    if truthfulness_head_address is not None:
        paths['truthfulness_head'] = to_path(truthfulness_head_address)
    else:
        # Search for truthfulness head files
        # Check in truthfulness_head subdirectory first
        truthfulness_dir = base_dir.parent / "truthfulness_head"
        if truthfulness_dir.exists():
            truthfulness_pattern = f"truthfulness_head_{model_version}_avg.npy"
            truthfulness_files = list(truthfulness_dir.glob(truthfulness_pattern))
            if truthfulness_files:
                paths['truthfulness_head'] = truthfulness_files[0]
            else:
                paths['truthfulness_head'] = None
        else:
            paths['truthfulness_head'] = None
        
        # Also check in base_dir
        if paths['truthfulness_head'] is None:
            truthfulness_pattern = f"truthfulness_head_{model_version}_avg.npy"
            truthfulness_files = list(base_dir.glob(truthfulness_pattern))
            if truthfulness_files:
                paths['truthfulness_head'] = truthfulness_files[0]
    
    return paths


def load_retrieval_heads_from_json(json_path: Path) -> List[Tuple[int, int, float]]:
    """
    Load retrieval heads from JSON file.
    
    Args:
        json_path: Path to JSON file containing retrieval head scores
        
    Returns:
        List of (layer, head, score) tuples for retrieval heads, sorted by average score (descending)
    """
    with open(json_path, 'r') as file:
        head_list = json.loads(file.readline())
    
    # Check if file is empty
    if not head_list or (isinstance(head_list, dict) and len(head_list) == 0):
        print(f"⚠ Warning: Retrieval head file is empty: {json_path}")
        return []
    
    # Use the average retrieval score and ranking
    # Format: [([layer, head], mean_score), ...]
    head_score_list = [([int(ll) for ll in l[0].split("-")], np.mean(l[1])) for l in head_list.items()]
    head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)
    
    # Extract (layer, head, score) tuples with scores
    heads_list = [(layer, head, score) for [layer, head], score in head_score_list]
    
    return heads_list


def visualize_heads(
    head_type: Union[str, List[str]],
    model_name: str,
    base_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    top_k: int = 30,
    n_layers: int = 32,
    n_heads: int = 32,
    inv_threshold: float = 0.7
) -> Optional[Path]:
    """
    Visualize head detection results.
    
    Args:
        head_type: Head type(s) to visualize. Can be:
            - Single type: "previous_token_head", "duplicate_token_head", "induction_head",
                          "retrieval_head", "iteration_head", "truthfulness_head"
            - "all": Visualize all head types in one combined plot
            - List of types: Visualize specified types in one combined plot
        model_name: Model name, e.g., "meta-llama/Meta-Llama-3-8B-Instruct"
        base_dir: Base directory for head scores. If None, uses default location.
        output_path: Path to save visualization. If None, saves to default location.
        top_k: Number of top heads to highlight (default: 30)
        n_layers: Number of layers in the model (default: 32)
        n_heads: Number of heads per layer (default: 32)
        inv_threshold: Invariance threshold for iteration heads (default: 0.7)
        
    Returns:
        Path to saved visualization file, or None if visualization failed
    """
    model_version = model_name.split("/")[-1]
    
    # Get file paths
    file_paths = get_head_score_paths(model_name, base_dir)
    
    # Handle "all" case
    if head_type == "all":
        head_type = ["previous_token_head", "duplicate_token_head", "induction_head",
                     "retrieval_head", "iteration_head", "truthfulness_head"]
    
    # Convert single type to list
    if isinstance(head_type, str):
        head_type = [head_type]
    
    # Determine if we need combined visualization
    need_combined = len(head_type) > 1
    
    if need_combined:
        # Combined visualization
        print("="*60)
        print(f"Loading head scores for combined visualization...")
        print(f"Model: {model_name}")
        print(f"Head types: {', '.join(head_type)}")
        print("="*60)
        
        # Load all required head types
        previous_scores = None
        duplicate_scores = None
        induction_scores = None
        truthfulness_scores = None
        truthfulness_heads_list = None
        retrieval_heads_list = None
        iteration_scores = None
        
        # Load pattern-based heads
        if "previous_token_head" in head_type:
            if file_paths['previous_token_head'].exists():
                previous_scores = load_head_scores(file_paths['previous_token_head'])
                print(f"✓ Loaded Previous Token Head: shape {previous_scores.shape}")
                n_layers, n_heads = previous_scores.shape
                # 打印前30个
                flat_scores = previous_scores.flatten()
                top_k_values, top_k_indices = torch.topk(flat_scores, k=min(30, flat_scores.numel()), largest=True)
                print(f"  前30个 Previous Token Heads (从大到小):")
                for i, idx in enumerate(top_k_indices, 1):
                    layer = idx.item() // n_heads
                    head = idx.item() % n_heads
                    score = previous_scores[layer, head].item()
                    print(f"    {i:2d}. Layer {layer:2d}, Head {head:2d}, Score: {score:.6f}")
            else:
                print(f"⚠ File not found: {file_paths['previous_token_head']}")
        
        if "duplicate_token_head" in head_type:
            if file_paths['duplicate_token_head'].exists():
                duplicate_scores = load_head_scores(file_paths['duplicate_token_head'])
                print(f"✓ Loaded Duplicate Token Head: shape {duplicate_scores.shape}")
                if previous_scores is None:
                    n_layers, n_heads = duplicate_scores.shape
                # 打印前30个
                flat_scores = duplicate_scores.flatten()
                top_k_values, top_k_indices = torch.topk(flat_scores, k=min(30, flat_scores.numel()), largest=True)
                print(f"  前30个 Duplicate Token Heads (从大到小):")
                for i, idx in enumerate(top_k_indices, 1):
                    layer = idx.item() // n_heads
                    head = idx.item() % n_heads
                    score = duplicate_scores[layer, head].item()
                    print(f"    {i:2d}. Layer {layer:2d}, Head {head:2d}, Score: {score:.6f}")
            else:
                print(f"⚠ File not found: {file_paths['duplicate_token_head']}")
        
        if "induction_head" in head_type:
            if file_paths['induction_head'].exists():
                induction_scores = load_head_scores(file_paths['induction_head'])
                print(f"✓ Loaded Induction Head: shape {induction_scores.shape}")
                if previous_scores is None and duplicate_scores is None:
                    n_layers, n_heads = induction_scores.shape
                # 打印前30个
                flat_scores = induction_scores.flatten()
                top_k_values, top_k_indices = torch.topk(flat_scores, k=min(30, flat_scores.numel()), largest=True)
                print(f"  前30个 Induction Heads (从大到小):")
                for i, idx in enumerate(top_k_indices, 1):
                    layer = idx.item() // n_heads
                    head = idx.item() % n_heads
                    score = induction_scores[layer, head].item()
                    print(f"    {i:2d}. Layer {layer:2d}, Head {head:2d}, Score: {score:.6f}")
            else:
                print(f"⚠ File not found: {file_paths['induction_head']}")
        
        # Get shape reference - use first available pattern head, or default if none exist
        # 不能使用 or 操作符，因为 Tensor 不能直接转换为布尔值
        if previous_scores is not None:
            reference_scores = previous_scores
            n_layers, n_heads = reference_scores.shape
        elif duplicate_scores is not None:
            reference_scores = duplicate_scores
            n_layers, n_heads = reference_scores.shape
        elif induction_scores is not None:
            reference_scores = induction_scores
            n_layers, n_heads = reference_scores.shape
        else:
            # No pattern heads found, use default shape
            print("⚠ No pattern heads found, using default shape (32, 32)")
            n_layers, n_heads = 32, 32
        
        # Ensure all pattern heads have the same shape
        if previous_scores is not None and previous_scores.shape != (n_layers, n_heads):
            print(f"⚠ Warning: Previous head shape {previous_scores.shape} != reference {(n_layers, n_heads)}")
        if duplicate_scores is not None and duplicate_scores.shape != (n_layers, n_heads):
            print(f"⚠ Warning: Duplicate head shape {duplicate_scores.shape} != reference {(n_layers, n_heads)}")
        if induction_scores is not None and induction_scores.shape != (n_layers, n_heads):
            print(f"⚠ Warning: Induction head shape {induction_scores.shape} != reference {(n_layers, n_heads)}")
        
        # Load truthfulness heads
        if "truthfulness_head" in head_type:
            if file_paths['truthfulness_head'] is not None and file_paths['truthfulness_head'].exists():
                # Try loading as npy first (averaged heads)
                try:
                    heads_array = np.load(file_paths['truthfulness_head'])
                    if heads_array.size > 0:
                        truthfulness_heads_list = [(int(row[0]), int(row[1])) for row in heads_array]
                        # Create scores matrix
                        truthfulness_scores = torch.zeros(n_layers, n_heads)
                        for layer, head in truthfulness_heads_list:
                            if layer < n_layers and head < n_heads:
                                truthfulness_scores[layer, head] = 1.0
                        print(f"✓ Loaded Truthfulness Heads: {len(truthfulness_heads_list)} heads")
                        # 打印前30个
                        print(f"  前30个 Truthfulness Heads:")
                        for i, (layer, head) in enumerate(truthfulness_heads_list[:30], 1):
                            print(f"    {i:2d}. Layer {layer:2d}, Head {head:2d}")
                except:
                    # Try loading as JSON
                    try:
                        truthfulness_scores, truthfulness_heads_list = load_truthfulness_heads(
                            file_paths['truthfulness_head'], n_layers=n_layers, n_heads=n_heads
                        )
                        print(f"✓ Loaded Truthfulness Heads: {len(truthfulness_heads_list)} heads")
                        # 打印前30个
                        print(f"  前30个 Truthfulness Heads:")
                        for i, (layer, head) in enumerate(truthfulness_heads_list[:30], 1):
                            print(f"    {i:2d}. Layer {layer:2d}, Head {head:2d}")
                    except Exception as e:
                        print(f"⚠ Could not load truthfulness heads: {e}, using zero matrix")
                        truthfulness_scores = torch.zeros(n_layers, n_heads)
            else:
                print(f"⚠ Truthfulness heads file not found, using zero matrix")
                truthfulness_scores = torch.zeros(n_layers, n_heads)
        
        # Load retrieval heads
        if "retrieval_head" in head_type:
            if file_paths['retrieval_head'] is not None and file_paths['retrieval_head'].exists():
                retrieval_heads_list = load_retrieval_heads_from_json(file_paths['retrieval_head'])

                print(f"✓ Loaded Retrieval Heads: {len(retrieval_heads_list)} heads")
                # 打印前30个（包含分数）
                print(f"  前30个 Retrieval Heads (从大到小):")
                for i, (layer, head, score) in enumerate(retrieval_heads_list[:30], 1):
                    print(f"    {i:2d}. Layer {layer:2d}, Head {head:2d}, Score: {score:.6f}")
            else:
                print(f"⚠ Retrieval heads file not found, using empty list")
                retrieval_heads_list = []
        
        # Load iteration heads
        if "iteration_head" in head_type:
            if file_paths['iteration_head'] is not None and file_paths['iteration_head'].exists():
                iteration_scores = load_iteration_heads(
                    file_paths['iteration_head'], n_layers=n_layers, n_heads=n_heads
                )
                print(f"✓ Loaded Iteration Heads")
                # 打印前30个
                flat_scores = iteration_scores.flatten()
                top_k_values, top_k_indices = torch.topk(flat_scores, k=min(30, flat_scores.numel()), largest=True)
                print(f"  前30个 Iteration Heads (从大到小):")
                for i, idx in enumerate(top_k_indices, 1):
                    layer = idx.item() // n_heads
                    head = idx.item() % n_heads
                    score = iteration_scores[layer, head].item()
                    print(f"    {i:2d}. Layer {layer:2d}, Head {head:2d}, Score: {score:.6f}")
            else:
                print(f"⚠ Iteration heads file not found, using zero matrix")
                iteration_scores = torch.zeros(n_layers, n_heads)
        
        # Create combined visualization
        print("\n" + "="*60)
        print("Creating combined visualization with vertical splits...")
        print("="*60)
        
        # Set default output path
        if output_path is None:
            if base_dir is None:
                current_file = Path(__file__)
                retrieval_head_dir = current_file.parent.parent
                base_dir = retrieval_head_dir / "head_score_all" / model_version
            output_dir = base_dir / "visualization"
            output_dir.mkdir(parents=True, exist_ok=True)
            head_types_str = "_".join(sorted(head_type))
            output_path = output_dir / f"combined_{head_types_str}_top{top_k}.png"
        
        # 确保所有 scores 都是 Tensor，如果为 None 则使用零矩阵
        # 不能使用 or 操作符，因为 Tensor 不能直接转换为布尔值
        prev_scores = previous_scores if previous_scores is not None else torch.zeros(n_layers, n_heads)
        dup_scores = duplicate_scores if duplicate_scores is not None else torch.zeros(n_layers, n_heads)
        ind_scores = induction_scores if induction_scores is not None else torch.zeros(n_layers, n_heads)
        # 打印所有变量的长度信息

        # Extract (layer, head) tuples for visualization (retrieval_heads_list contains scores)
        retrieval_heads_list_for_viz = None
        if retrieval_heads_list is not None and len(retrieval_heads_list) > 0:
            retrieval_heads_list_for_viz = [(layer, head) for layer, head, score in retrieval_heads_list]
        
        visualize_combined_heads(
            prev_scores,
            dup_scores,
            ind_scores,
            truthfulness_scores=truthfulness_scores,
            truthfulness_heads_list=truthfulness_heads_list,
            retrieval_heads_list=retrieval_heads_list_for_viz,
            iteration_scores=iteration_scores,
            top_k=top_k,
            save_path=output_path
        )
        
        print(f"\n✓ Visualization saved to: {output_path}")
        return output_path
        
    else:
        # Single head type visualization
        head_type_str = head_type[0]
        print("="*60)
        print(f"Visualizing {head_type_str}")
        print(f"Model: {model_name}")
        print("="*60)
        
        file_path = file_paths.get(head_type_str)
        
        # Load scores based on file type, use zero matrix if file not found
        if file_path is None or not file_path.exists():
            print(f"⚠ File not found for {head_type_str}, using zero matrix")
            print(f"  Expected path: {file_path}")
            scores = torch.zeros(n_layers, n_heads)
            title = head_type_str.replace("_", " ").title()
        elif head_type_str in ["previous_token_head", "duplicate_token_head", "induction_head"]:
            scores = load_head_scores(file_path)
            title = head_type_str.replace("_", " ").title()
        elif head_type_str == "iteration_head":
            scores = load_iteration_heads(file_path, n_layers=n_layers, n_heads=n_heads)
            title = "Iteration Head"
        elif head_type_str == "truthfulness_head":
            # Load as npy or json
            try:
                heads_array = np.load(file_path)
                if heads_array.size > 0:
                    heads_list = [(int(row[0]), int(row[1])) for row in heads_array]
                    scores = torch.zeros(n_layers, n_heads)
                    for layer, head in heads_list:
                        if layer < n_layers and head < n_heads:
                            scores[layer, head] = 1.0
                else:
                    scores = torch.zeros(n_layers, n_heads)
            except:
                try:
                    scores, _ = load_truthfulness_heads(file_path, n_layers=n_layers, n_heads=n_heads)
                except:
                    print(f"⚠ Could not load truthfulness heads, using zero matrix")
                    scores = torch.zeros(n_layers, n_heads)
            title = "Truthfulness Head"
        elif head_type_str == "retrieval_head":
            # For retrieval head, we need to create a scores matrix from the list
            try:
                heads_list = load_retrieval_heads_from_json(file_path)
                scores = torch.zeros(n_layers, n_heads)
                for layer, head, score in heads_list:
                    if layer < n_layers and head < n_heads:
                        scores[layer, head] = score
            except:
                print(f"⚠ Could not load retrieval heads, using zero matrix")
                scores = torch.zeros(n_layers, n_heads)
            title = "Retrieval Head"
        else:
            print(f"❌ Error: Unknown head type: {head_type_str}")
            return None
        
        # Set default output path
        if output_path is None:
            if base_dir is None:
                current_file = Path(__file__)
                retrieval_head_dir = current_file.parent.parent
                base_dir = retrieval_head_dir / "head_score_all" / model_version
            output_dir = base_dir / "visualization"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{head_type_str}_top{top_k}.png"
        
        # Visualize
        visualize_head_scores(scores, title, top_k=top_k, save_path=output_path)
        
        print(f"\n✓ Visualization saved to: {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize head detection results')
    parser.add_argument('--head_type', type=str, default='all',
                        help='Head type(s) to visualize: "all", "previous_token_head", "duplicate_token_head", "induction_head", "retrieval_head", "iteration_head", "truthfulness_head", or comma-separated list')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Model name, e.g., "meta-llama/Meta-Llama-3-8B-Instruct"')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Base directory for head scores. If None, uses default location.')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save visualization. If None, saves to default location.')
    parser.add_argument('--top_k', type=int, default=30,
                        help='Number of top heads to highlight (default: 30)')
    parser.add_argument('--n_layers', type=int, default=32,
                        help='Number of layers in the model (default: 32)')
    parser.add_argument('--n_heads', type=int, default=32,
                        help='Number of heads per layer (default: 32)')
    parser.add_argument('--inv_threshold', type=float, default=0.7,
                        help='Invariance threshold for iteration heads (default: 0.7)')
    
    args = parser.parse_args()
    
    # Parse head_type (support comma-separated list)
    if ',' in args.head_type:
        head_type = [h.strip() for h in args.head_type.split(',')]
    else:
        head_type = args.head_type
    
    # Convert paths
    base_dir = Path(args.base_dir) if args.base_dir else None
    output_path = Path(args.output_path) if args.output_path else None
    
    # ============================================
    # 地址设置说明：
    # ============================================
    # 1. base_dir: 设置 head score 文件的输入目录（从哪里读取文件）
    #    如果不设置，默认使用: Retrieval_Head/head_score_all/{model_version}/
    #
    # 2. output_path: 设置可视化结果的保存路径（保存到哪里）
    #    如果不设置，默认保存到: {base_dir}/visualization/{filename}.png
    #
    # 3. top_k: 可视化的 top k 值，可以通过命令行参数 --top_k 设置
    # ============================================
    
    print("="*60)
    print("Head Visualization")
    print("="*60)
    print(f"Head type: {head_type}")
    print(f"Model: {args.model_name}")
    print(f"Top K: {args.top_k}")
    print(f"Layers: {args.n_layers}, Heads per layer: {args.n_heads}")
    if base_dir:
        print(f"Base directory: {base_dir}")
    if output_path:
        print(f"Output path: {output_path}")
    print("="*60)
    
    visualize_heads(
        head_type=head_type,
        model_name=args.model_name,
        base_dir=base_dir,
        output_path=output_path,
        top_k=args.top_k,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        inv_threshold=args.inv_threshold
    )
    
    # # Example 2: 自定义输入目录（base_dir）
    # visualize_heads(
    #     head_type="all",
    #     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    #     base_dir=Path("/path/to/your/head_scores/Meta-Llama-3-8B-Instruct")
    # )
    
    # # Example 3: 自定义输出路径（output_path）
    # visualize_heads(
    #     head_type="iteration_head",
    #     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    #     output_path=Path("/path/to/save/visualization.png")
    # )
    # path = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Llama-2-7b-hf/Llama-2-7b-hf_retrieval_head.json"
    # path ="/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_retrieval_head.json"
    path = "/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_retrieval_head.json"
    # with open(f"{path}", "r") as file:
    #     stable_block_list =  json.loads(file.readline())
    # stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()]
    # stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True) 
    # block_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list][:100]
    # print(block_list)

    with open(f'{path}') as file:
        head_list = json.loads(file.readline())
    ## use the average retrieval score and ranking
    head_score_list = [([int(ll) for ll in l[0].split("-")],np.mean(l[1])) for l in head_list.items()]
    head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True) 
    top_retrieval_heads = [[l[0],  round(np.mean(l[1]), 2)] for l in head_score_list][:10]
    print(top_retrieval_heads)
    exit()
    # Load and print the file
    print("\n" + "="*60)
    print(f"Loading and printing: {path}")
    print("="*60)
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        print(f"\nFile loaded successfully!")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Number of entries: {len(data)}")
            
            if len(data) == 0:
                print("⚠ File is empty (no data)")
            else:
                # Calculate statistics
                total_scores = 0
                head_scores_list = []
                
                for head_key, scores in data.items():
                    if isinstance(scores, list):
                        total_scores += len(scores)
                        if len(scores) > 0:
                            mean_score = np.mean(scores)
                            layer_idx, head_idx = map(int, head_key.split("-"))
                            head_scores_list.append((layer_idx, head_idx, mean_score, len(scores)))
                
                # Sort by mean score
                head_scores_list.sort(key=lambda x: x[2], reverse=True)
                
                print(f"Total score entries: {total_scores}")
                print(f"Heads with scores: {len(head_scores_list)}")
                
                if len(head_scores_list) > 0:
                    print(f"\nTop 30 heads by mean score:")
                    for i, (layer, head, mean_score, count) in enumerate(head_scores_list[:30], 1):
                        print(f"  {i:2d}. Layer {layer:2d}, Head {head:2d}: mean={mean_score:.6f}, count={count}")
                    
                    if len(head_scores_list) > 30:
                        print(f"\nLast 10 heads:")
                        for i, (layer, head, mean_score, count) in enumerate(head_scores_list[-10:], len(head_scores_list) - 9):
                            print(f"  {i:2d}. Layer {layer:2d}, Head {head:2d}: mean={mean_score:.6f}, count={count}")
                else:
                    print("⚠ No heads with valid scores found")
        elif isinstance(data, list):
            print(f"Number of items: {len(data)}")
            if len(data) > 0:
                print(f"\nFirst 10 items:")
                for i, item in enumerate(data[:10], 1):
                    print(f"  {i}. {item}")
        else:
            print(f"Data: {data}")
            
    except FileNotFoundError:
        print(f"❌ Error: File not found: {path}")
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {e}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
    
    print("\n" + "="*60)
    
    
    # # Example 4: 同时自定义输入和输出地址
    # visualize_heads(
    #     head_type="all",
    #     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    #     base_dir=Path("/path/to/head_scores/Meta-Llama-3-8B-Instruct"),
    #     output_path=Path("/path/to/output/combined_heads.png")
    # )

