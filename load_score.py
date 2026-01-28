"""
Load head scores from various file formats.

This module provides functions to load head scores from different file formats:
- JSON files: Dictionary with 'layer-head' keys and score lists
- NPY files: Arrays with [layer, head] or [layer, head, score] format
- PT files: PyTorch tensors with shape (n_layers, n_heads)
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union
import torch


def load_head_score(score_path: Union[str, Path], head_type: str = None, 
                    n_layers: int = 32, n_heads: int = 32) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Load head scores from a file and return scores matrix and head list.
    
    Args:
        score_path: Path to the score file (JSON, NPY, or PT format)
        head_type: Type of head (optional, used for special handling):
                   - 'truthfulness': For truthfulness head files
                   - 'iteration': For iteration head files
                   - 'induction': For induction head files
                   - 'duplicate': For duplicate token head files
                   - 'previous': For previous token head files
                   - 'retrieval': For retrieval head files
                   - None: Auto-detect based on file extension
        n_layers: Number of layers in the model (default: 32)
        n_heads: Number of heads per layer (default: 32)
        
    Returns:
        Tuple of (scores, head_list):
        - scores: torch.Tensor of shape (n_layers, n_heads) containing head scores
        - head_list: List of (layer, head) tuples for heads with non-zero scores
    """
    score_path = Path(score_path)
    
    if not score_path.exists():
        raise FileNotFoundError(f"Score file not found: {score_path}")
    
    file_ext = score_path.suffix.lower()
    
    # Auto-detect head_type from filename if not provided
    if head_type is None:
        filename = score_path.name.lower()
        if 'truthfulness' in filename:
            head_type = 'truthfulness'
        elif 'iteration' in filename:
            head_type = 'iteration'
        elif 'induction' in filename:
            head_type = 'induction'
        elif 'duplicate' in filename:
            head_type = 'duplicate'
        elif 'previous' in filename:
            head_type = 'previous'
        elif 'retrieval' in filename:
            head_type = 'retrieval'
    
    # Load based on file extension
    if file_ext == '.json':
        scores, head_list = _load_from_json(score_path, n_layers, n_heads)
    elif file_ext == '.npy':
        scores, head_list = _load_from_npy(score_path, head_type, n_layers, n_heads)
    elif file_ext == '.pt':
        scores, head_list = _load_from_pt(score_path, n_layers, n_heads)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .json, .npy, .pt")
    
    return scores, head_list


def _load_from_json(score_path: Path, n_layers: int, n_heads: int) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Load scores from JSON file."""
    with open(score_path, 'r') as f:
        data = json.load(f)
    
    # Initialize scores matrix
    scores = torch.zeros(n_layers, n_heads)
    head_list = []
    
    if isinstance(data, dict):
        # Format: {'layer-head': [score_list]} or {'layer-head': score}
        for key, score_value in data.items():
            try:
                layer, head = map(int, key.split('-'))
                # Handle both list and single value
                if isinstance(score_value, list):
                    # Use mean of score list as the score for this head
                    if score_value and len(score_value) > 0:
                        score = float(np.mean(score_value))
                    else:
                        score = 0.0
                else:
                    # Single value
                    score = float(score_value)
                
                if layer < n_layers and head < n_heads:
                    scores[layer, head] = score
                    if score > 0:
                        head_list.append((layer, head))
            except (ValueError, IndexError, TypeError):
                continue
    elif isinstance(data, list):
        # Format: [{'layer': int, 'head': int, 'score': float}, ...]
        for item in data:
            if isinstance(item, dict) and 'layer' in item and 'head' in item:
                layer = int(item['layer'])
                head = int(item['head'])
                score = float(item.get('score', 1.0))
                if layer < n_layers and head < n_heads:
                    scores[layer, head] = score
                    if score > 0:
                        head_list.append((layer, head))
    
    return scores, head_list


def _load_from_npy(score_path: Path, head_type: str, n_layers: int, n_heads: int) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Load scores from NPY file."""
    heads_array = np.load(score_path, allow_pickle=True)
    
    if heads_array.size == 0:
        return torch.zeros(n_layers, n_heads), []
    
    # Handle different array shapes
    if heads_array.ndim == 1:
        heads_array = heads_array.reshape(1, -1)
    
    scores = torch.zeros(n_layers, n_heads)
    head_list = []
    
    # Check number of columns
    if heads_array.shape[1] >= 3:
        # Format: [layer, head, score]
        for row in heads_array:
            layer = int(row[0])
            head = int(row[1])
            score = float(row[2])
            if layer < n_layers and head < n_heads:
                scores[layer, head] = score
                if score > 0:
                    head_list.append((layer, head))
    elif heads_array.shape[1] >= 2:
        # Format: [layer, head] - assign score of 1.0
        for row in heads_array:
            layer = int(row[0])
            head = int(row[1])
            if layer < n_layers and head < n_heads:
                scores[layer, head] = 1.0
                head_list.append((layer, head))
    
    return scores, head_list


def _load_from_pt(score_path: Path, n_layers: int, n_heads: int) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Load scores from PyTorch PT file."""
    scores = torch.load(score_path, map_location='cpu')
    
    # Ensure it's a tensor
    if not isinstance(scores, torch.Tensor):
        raise ValueError(f"PT file does not contain a torch.Tensor, got {type(scores)}")
    
    # Handle different shapes
    if scores.ndim == 2:
        # Shape: (n_layers, n_heads)
        if scores.shape[0] != n_layers or scores.shape[1] != n_heads:
            # Resize or crop if needed
            if scores.shape[0] > n_layers or scores.shape[1] > n_heads:
                scores = scores[:n_layers, :n_heads]
            else:
                # Pad with zeros
                new_scores = torch.zeros(n_layers, n_heads)
                new_scores[:scores.shape[0], :scores.shape[1]] = scores
                scores = new_scores
    else:
        raise ValueError(f"Unexpected tensor shape: {scores.shape}, expected (n_layers, n_heads)")
    
    # Extract head list (non-zero scores)
    head_list = []
    for layer in range(n_layers):
        for head in range(n_heads):
            if scores[layer, head] > 0:
                head_list.append((layer, head))
    
    return scores, head_list


# Example usage
if __name__ == "__main__":
    # Example: Load different types of head scores
    base_dir = Path("head_score_all/Llama-2-7b-hf")
    
    # Example 1: Load truthfulness heads
    truthfulness_path = base_dir / "truthfulness_head_Llama-2-7b-hf_avg.npy"
    if truthfulness_path.exists():
        scores, head_list = load_head_score(truthfulness_path, head_type='truthfulness')
        print(f"Truthfulness heads: {len(head_list)} heads loaded")
        print(f"Score shape: {scores.shape}")
        non_zero_scores = scores[scores > 0]
        if len(non_zero_scores) > 0:
            print(f"Score range: [{non_zero_scores.min():.4f}, {non_zero_scores.max():.4f}]")
        else:
            print("Score range: No non-zero scores")
        print(f"First 5 heads with scores:")
        for layer, head in head_list[:5]:
            score = scores[layer, head].item()
            print(f"  Layer {layer}, Head {head}: {score:.4f}")
    
    # Example 2: Load iteration heads
    iteration_path = base_dir / "Llama-2-7b-hf_iteration_heads_inv_gt_0.70_sorted.npy"
    if iteration_path.exists():
        scores, head_list = load_head_score(iteration_path, head_type='iteration')
        print(f"\nIteration heads: {len(head_list)} heads loaded")
        print(f"Score shape: {scores.shape}")
        non_zero_scores = scores[scores > 0]
        if len(non_zero_scores) > 0:
            print(f"Score range: [{non_zero_scores.min():.4f}, {non_zero_scores.max():.4f}]")
        else:
            print("Score range: No non-zero scores")
        print(f"First 5 heads with scores:")
        for layer, head in head_list[:5]:
            score = scores[layer, head].item()
            print(f"  Layer {layer}, Head {head}: {score:.4f}")
    
    # Example 3: Load induction heads (PT file)
    induction_path = base_dir / "Llama-2-7b-hf_induction_head_custom_abs.pt"
    if induction_path.exists():
        scores, head_list = load_head_score(induction_path, head_type='induction')
        # Sort by score (descending order)
        head_list_sorted = sorted(head_list, key=lambda x: scores[x[0], x[1]].item(), reverse=True)
        print(f"\nInduction heads: {len(head_list_sorted)} heads loaded (sorted by score)")
        print(f"Score shape: {scores.shape}")
        non_zero_scores = scores[scores > 0]
        if len(non_zero_scores) > 0:
            print(f"Score range: [{non_zero_scores.min():.4f}, {non_zero_scores.max():.4f}]")
        else:
            print("Score range: No non-zero scores")
        print(f"Top 5 heads with scores (sorted by score, highest first):")
        for layer, head in head_list_sorted[:5]:
            score = scores[layer, head].item()
            print(f"  Layer {layer}, Head {head}: {score:.4f}")
    
    # Example 4: Load duplicate heads
    duplicate_path = base_dir / "Llama-2-7b-hf_duplicate_head.npy"
    if duplicate_path.exists():
        scores, head_list = load_head_score(duplicate_path, head_type='duplicate')
        print(f"\nDuplicate heads: {len(head_list)} heads loaded")
        print(f"Score shape: {scores.shape}")
        non_zero_scores = scores[scores > 0]
        if len(non_zero_scores) > 0:
            print(f"Score range: [{non_zero_scores.min():.4f}, {non_zero_scores.max():.4f}]")
        else:
            print("Score range: No non-zero scores")
        print(f"First 5 heads with scores:")
        for layer, head in head_list[:5]:
            score = scores[layer, head].item()
            print(f"  Layer {layer}, Head {head}: {score:.4f}")
    
    # Example 5: Load previous token heads
    previous_path = base_dir / "Llama-2-7b-hf_previous_head.npy"
    if previous_path.exists():
        scores, head_list = load_head_score(previous_path, head_type='previous')
        print(f"\nPrevious token heads: {len(head_list)} heads loaded")
        print(f"Score shape: {scores.shape}")
        non_zero_scores = scores[scores > 0]
        if len(non_zero_scores) > 0:
            print(f"Score range: [{non_zero_scores.min():.4f}, {non_zero_scores.max():.4f}]")
        else:
            print("Score range: No non-zero scores")
        print(f"First 5 heads with scores:")
        for layer, head in head_list[:5]:
            score = scores[layer, head].item()
            print(f"  Layer {layer}, Head {head}: {score:.4f}")
    
    # Example 6: Load retrieval heads
    retrieval_path = base_dir / "Llama-2-7b-hf_retrieval_head.json"
    if retrieval_path.exists():
        scores, head_list = load_head_score(retrieval_path, head_type='retrieval')
        # Sort by score (descending order)
        head_list_sorted = sorted(head_list, key=lambda x: scores[x[0], x[1]].item(), reverse=True)
        print(f"\nRetrieval heads: {len(head_list_sorted)} heads loaded (sorted by score)")
        print(f"Score shape: {scores.shape}")
        non_zero_scores = scores[scores > 0]
        if len(non_zero_scores) > 0:
            print(f"Score range: [{non_zero_scores.min():.4f}, {non_zero_scores.max():.4f}]")
        else:
            print("Score range: No non-zero scores")
        print(f"Top 5 heads with scores (sorted by score, highest first):")
        for layer, head in head_list_sorted[:5]:
            score = scores[layer, head].item()
            print(f"  Layer {layer}, Head {head}: {score:.4f}")

