"""
Visualize top 30 heads from head detection scores with vertical splits.

This script loads head score files and visualizes the top 30 heads
for each detection pattern type. Supports 5 head types:
- Previous Token Head
- Duplicate Token Head  
- Induction Head
- Truthfulness Head
- Retrieval Head

Uses vertical splits to clearly show overlapping head functions.
"""

import torch
import numpy as np
import json
from pathlib import Path

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


def load_head_scores(file_path: Path) -> torch.Tensor:
    """Load head scores from a .pt file.
    
    Args:
        file_path: Path to the .pt file
        
    Returns:
        Tensor of shape (n_layers, n_heads) containing head scores
    """
    scores = torch.load(file_path, map_location='cpu')
    return scores


def load_truthfulness_heads(json_path: Path, n_layers: int = 32, n_heads: int = 32) -> tuple:
    """Load truthfulness heads from JSON or NPY file and create scores matrix.
    
    Args:
        json_path: Path to the truthfulness heads file (JSON or .npy)
        n_layers: Number of layers in the model
        n_heads: Number of heads per layer
        
    Returns:
        Tuple of (scores_matrix, heads_list)
        - scores_matrix: Tensor of shape (n_layers, n_heads) with scores (0 for non-truthfulness heads)
        - heads_list: List of (layer, head) tuples for truthfulness heads
    """
    # Check file extension to determine format
    if json_path.suffix == '.npy':
        # Load from .npy file (format: array of [layer, head] or [layer, head, score])
        heads_array = np.load(json_path)
        if heads_array.size == 0:
            print(f"  ⚠ Warning: Empty truthfulness heads file: {json_path.name}")
            return torch.zeros(n_layers, n_heads), []
        
        # Handle different array shapes
        if heads_array.ndim == 1:
            heads_array = heads_array.reshape(1, -1)
        
        # Extract heads (and scores if available)
        heads_list = []
        scores_dict = {}
        for row in heads_array:
            if len(row) >= 2:
                layer, head = int(row[0]), int(row[1])
                heads_list.append((layer, head))
                # If score is available (3rd column), use it; otherwise use 1.0
                if len(row) >= 3:
                    score = float(row[2])
                else:
                    score = 1.0
                scores_dict[(layer, head)] = score
        
        # Create scores matrix
        scores_matrix = torch.zeros(n_layers, n_heads)
        for (layer, head), score in scores_dict.items():
            if layer < n_layers and head < n_heads:
                scores_matrix[layer, head] = score
        
        print(f"  Loaded {len(heads_list)} truthfulness heads from {json_path.name}")
        if scores_dict:
            print(f"  Score range: [{min(scores_dict.values()):.4f}, {max(scores_dict.values()):.4f}]")
        
        return scores_matrix, heads_list
    else:
        # Load from JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract heads information
        heads_list = [(head_info['layer'], head_info['head']) for head_info in data['heads']]
        scores_dict = {(head_info['layer'], head_info['head']): head_info['score'] 
                       for head_info in data['heads']}
        
        # Create scores matrix
        scores_matrix = torch.zeros(n_layers, n_heads)
        for (layer, head), score in scores_dict.items():
            if layer < n_layers and head < n_heads:
                scores_matrix[layer, head] = score
        
        print(f"  Loaded {len(heads_list)} truthfulness heads from {json_path.name}")
        print(f"  Score range: [{min(scores_dict.values()):.4f}, {max(scores_dict.values()):.4f}]")
        
        return scores_matrix, heads_list


def load_iteration_heads(file_path: Path, n_layers: int = 32, n_heads: int = 32) -> torch.Tensor:
    """Load iteration heads from a .npy file and convert to score matrix.
    
    Args:
        file_path: Path to the .npy file containing head triplets [layer, head, peakiness]
        n_layers: Number of layers in the model
        n_heads: Number of heads per layer
        
    Returns:
        Tensor of shape (n_layers, n_heads) containing peakiness scores
        Only the heads in the file will have non-zero values
    """
    # Load numpy array: shape (n_heads_found, 3) where each row is [layer_idx, head_idx, peakiness]
    head_data = np.load(file_path)
    
    if head_data.size == 0:
        print(f"Warning: {file_path} is empty")
        return torch.zeros((n_layers, n_heads))
    
    # Create score matrix filled with zeros
    scores = torch.zeros((n_layers, n_heads))
    
    # Fill in the scores for each head
    for row in head_data:
        layer_idx = int(row[0])
        head_idx = int(row[1])
        peakiness = float(row[2])
        if layer_idx < n_layers and head_idx < n_heads:
            scores[layer_idx, head_idx] = peakiness
    
    n_iteration_heads = (scores > 0).sum().item()
    print(f"  Loaded {n_iteration_heads} iteration heads from {file_path.name}")
    if n_iteration_heads > 0:
        print(f"  Score range: [{scores[scores > 0].min().item():.4f}, {scores.max().item():.4f}]")
    
    return scores


def get_top_k_heads(scores: torch.Tensor, k: int = 30) -> tuple:
    """Get top k heads by score.
    
    Args:
        scores: Tensor of shape (n_layers, n_heads)
        k: Number of top heads to return
        
    Returns:
        Tuple of (top_scores, top_indices, top_layers, top_heads)
        - top_scores: Scores of top k heads
        - top_indices: Flat indices of top k heads
        - top_layers: Layer indices of top k heads
        - top_heads: Head indices of top k heads
    """
    # Flatten the scores and get top k indices
    flat_scores = scores.flatten()
    top_k_values, top_k_indices = torch.topk(flat_scores, k=k, largest=True)
    
    # Convert flat indices to (layer, head) pairs
    n_layers, n_heads = scores.shape
    top_layers = top_k_indices // n_heads
    top_heads = top_k_indices % n_heads
    
    return top_k_values, top_k_indices, top_layers, top_heads


def create_highlighted_matrix(scores: torch.Tensor, top_k: int = 30) -> torch.Tensor:
    """Create a matrix with top k heads highlighted.
    
    Args:
        scores: Tensor of shape (n_layers, n_heads)
        top_k: Number of top heads to highlight
        
    Returns:
        Matrix of same shape as scores, with top k heads kept and others set to min value
    """
    # Get top k heads
    _, _, top_layers, top_heads = get_top_k_heads(scores, k=top_k)
    
    # Create a new matrix with minimum values
    highlighted = torch.full_like(scores, scores.min())
    
    # Set top k heads to their original scores
    for layer, head in zip(top_layers, top_heads):
        highlighted[layer, head] = scores[layer, head]
    
    return highlighted


def visualize_with_neel_plotly(scores: torch.Tensor, title: str, top_k: int = 30, save_path: Path = None):
    """Visualize scores using neel_plotly.
    
    Args:
        scores: Tensor of shape (n_layers, n_heads)
        title: Plot title
        top_k: Number of top heads to highlight
        save_path: Optional path to save the figure
    """
    # Create highlighted matrix
    highlighted = create_highlighted_matrix(scores, top_k=top_k)
    
    # Transpose matrix: columns (x-axis) = layers, rows (y-axis) = heads
    highlighted = highlighted.T  # Shape becomes (n_heads, n_layers)
    
    # Get score range for color scale
    score_min = scores.min().item()
    score_max = scores.max().item()
    
    n_layers, n_heads = scores.shape
    
    # Create figure using plotly directly for better control
    # x-axis = layers (columns), y-axis = heads (rows)
    # Use custom colorscale with white for low values
    highlighted_np = highlighted.numpy()
    
    # Create colorscale that maps minimum values to white
    # Use white for low values, then transition to Viridis colors
    if score_max > score_min:
        # White for low values, then Viridis colors for higher values
        colorscale = [
            [0.0, 'rgb(255, 255, 255)'],  # Pure white for minimum values
            [0.25, 'rgb(255, 255, 255)'],  # Keep white for a larger range (0-25%)
            [0.3, 'rgb(68, 1, 84)'],      # Dark purple (Viridis start)
            [0.5, 'rgb(59, 82, 139)'],     # Blue
            [0.7, 'rgb(33, 144, 140)'],    # Teal
            [0.9, 'rgb(120, 198, 121)'],   # Light green
            [1.0, 'rgb(253, 231, 37)']     # Yellow (Viridis end)
        ]
    else:
        # If all values are the same, just use white
        colorscale = [[0.0, 'rgb(255, 255, 255)'], [1.0, 'rgb(255, 255, 255)']]
    
    fig = go.Figure(data=go.Heatmap(
        z=highlighted_np,
        x=list(range(n_layers)),  # Layer indices (x-axis, columns)
        y=list(range(n_heads)),  # Head indices (y-axis, rows)
        zmin=score_min,
        zmax=score_max,
        colorscale=colorscale,
        colorbar=dict(title="Score"),
        hovertemplate='Head: %{y}<br>Layer: %{x}<br>Score: %{z}<extra></extra>',
        showscale=True,
        # Add explicit cell boundaries with larger gaps
        xgap=2,  # Larger gap between columns
        ygap=2,  # Larger gap between rows
    ))
    
    # Add vertical lines between columns (layers)
    for i in range(n_layers + 1):
        fig.add_shape(
            type="line",
            x0=i - 0.5, x1=i - 0.5,
            y0=-0.5, y1=n_heads - 0.5,
            line=dict(color="black", width=2),
            layer="above"
        )
    
    # Add horizontal lines between rows (heads)
    for i in range(n_heads + 1):
        fig.add_shape(
            type="line",
            x0=-0.5, x1=n_layers - 0.5,
            y0=i - 0.5, y1=i - 0.5,
            line=dict(color="black", width=2),
            layer="above"
        )
    
    # Update layout to make cells clearly visible
    fig.update_layout(
        title=dict(
            text=f"{title} (Top {top_k} Heads Highlighted)",
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title="Layer",
            tickmode='linear',
            tick0=0,
            dtick=1,
            showgrid=False,  # Disable default grid, we use shapes instead
            showline=True,
            linewidth=2,
            linecolor='black',
            range=[-0.5, n_layers - 0.5]
        ),
        yaxis=dict(
            title="Head",
            tickmode='linear',
            tick0=0,
            dtick=1,
            showgrid=False,  # Disable default grid, we use shapes instead
            autorange='reversed',  # Head 0 at top
            showline=True,
            linewidth=2,
            linecolor='black',
            range=[n_heads - 0.5, -0.5]
        ),
        width=1000,
        height=800,
        plot_bgcolor='white'  # White background
    )
    
    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Try to save as PNG first if kaleido is available
        try:
            fig.write_image(str(save_path.with_suffix('.png')), width=1400, height=1200, scale=2)
            print(f"  Saved PNG: {save_path.with_suffix('.png')}")
        except Exception as e:
            # Fallback to HTML if PNG export fails
            fig.write_html(str(save_path.with_suffix('.html')))
            print(f"  Saved HTML: {save_path.with_suffix('.html')} (PNG export failed: {e})")
    else:
        # Show interactively
        fig.show()


def visualize_with_matplotlib(scores: torch.Tensor, title: str, top_k: int = 30, save_path: Path = None):
    """Visualize scores using matplotlib as fallback.
    
    Args:
        scores: Tensor of shape (n_layers, n_heads)
        title: Plot title
        top_k: Number of top heads to highlight
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create highlighted matrix
    highlighted = create_highlighted_matrix(scores, top_k=top_k)
    
    # Transpose matrix: columns (x-axis) = layers, rows (y-axis) = heads
    highlighted = highlighted.T  # Shape becomes (n_heads, n_layers)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    n_layers, n_heads = scores.shape
    
    # Use pcolormesh to ensure each head occupies a full cell
    # Create meshgrid for cell boundaries
    # x-axis = layers (columns), y-axis = heads (rows)
    x = np.arange(n_layers + 1) - 0.5  # Cell boundaries for x-axis (layers)
    y = np.arange(n_heads + 1) - 0.5  # Cell boundaries for y-axis (heads)
    X, Y = np.meshgrid(x, y)
    
    # Create a custom colormap with white base color
    # Start with pure white for low values, then use viridis colors
    # Use white for a large portion to ensure low scores are white
    colors = ['#ffffff', '#ffffff', '#ffffff', '#ffffff', '#440154', '#31688e', '#35b779', '#fde725']  # White + Viridis
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom_viridis', colors, N=n_bins)
    
    # Plot using pcolormesh - each cell represents one head
    # Use thicker black borders for clear separation
    im = ax.pcolormesh(X, Y, highlighted.numpy(), cmap=cmap, 
                       shading='flat', edgecolors='black', linewidths=2.0)  # Thicker black borders
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20)
    
    # Set labels and ticks
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Head', fontsize=14, fontweight='bold')
    ax.set_title(f"{title} (Top {top_k} Heads Highlighted)", fontsize=16, pad=20, fontweight='bold')
    
    # Set ticks to be at the center of each cell (integer positions)
    ax.set_xticks(range(n_layers))  # x-axis: layers
    ax.set_yticks(range(n_heads))  # y-axis: heads
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.5, n_heads - 0.5)
    
    # Invert y-axis so head 0 is at the top
    ax.invert_yaxis()
    
    # Add explicit grid lines for every row and column
    # Vertical lines (between layers/columns)
    for i in range(n_layers + 1):
        ax.axvline(x=i - 0.5, color='black', linewidth=2, alpha=1.0)
    
    # Horizontal lines (between heads/rows)
    for i in range(n_heads + 1):
        ax.axhline(y=i - 0.5, color='black', linewidth=2, alpha=1.0)
    
    ax.set_axisbelow(False)
    
    # Set background color to white
    ax.set_facecolor('white')  # White background
    fig.patch.set_facecolor('white')  # Also set figure background to white
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')  # White background
        print(f"  Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def decode_mask_value(mask_value: float) -> list:
    """Decode mask value to get list of head types.
    
    Encoding scheme:
    - 0: white (no type)
    - 1-6: single type (1=previous, 2=duplicate, 3=induction, 4=truthfulness, 5=retrieval, 6=iteration)
    - 10-99: 2 types (10 + type1*10 + type2)
    - 100-999: 3 types (100 + type1*100 + type2*10 + type3)
    - 1000-9999: 4 types (1000 + type1*1000 + type2*100 + type3*10 + type4)
    - 10000-99999: 5 types (10000 + type1*10000 + type2*1000 + type3*100 + type4*10 + type5)
    - 100000+: 6 types (100000 + type1*100000 + type2*10000 + type3*1000 + type4*100 + type5*10 + type6)
    
    Args:
        mask_value: Encoded mask value
        
    Returns:
        List of type integers, empty list if white
    """
    mask_value = int(mask_value)
    
    if mask_value == 0:
        return []
    elif 1 <= mask_value <= 6:
        return [mask_value]
    elif 10 <= mask_value < 100:
        # 2 types: 10 + type1*10 + type2
        remainder = mask_value - 10
        type1 = remainder // 10
        type2 = remainder % 10
        return [type1, type2]
    elif 100 <= mask_value < 1000:
        # 3 types: 100 + type1*100 + type2*10 + type3
        remainder = mask_value - 100
        type1 = remainder // 100
        remainder = remainder % 100
        type2 = remainder // 10
        type3 = remainder % 10
        return [type1, type2, type3]
    elif 1000 <= mask_value < 10000:
        # 4 types: 1000 + type1*1000 + type2*100 + type3*10 + type4
        remainder = mask_value - 1000
        type1 = remainder // 1000
        remainder = remainder % 1000
        type2 = remainder // 100
        remainder = remainder % 100
        type3 = remainder // 10
        type4 = remainder % 10
        return [type1, type2, type3, type4]
    elif 10000 <= mask_value < 100000:
        # 5 types: 10000 + type1*10000 + type2*1000 + type3*100 + type4*10 + type5
        remainder = mask_value - 10000
        type1 = remainder // 10000
        remainder = remainder % 10000
        type2 = remainder // 1000
        remainder = remainder % 1000
        type3 = remainder // 100
        remainder = remainder % 100
        type4 = remainder // 10
        type5 = remainder % 10
        return [type1, type2, type3, type4, type5]
    else:  # >= 100000
        # 6 types: 100000 + type1*100000 + type2*10000 + type3*1000 + type4*100 + type5*10 + type6
        remainder = mask_value - 100000
        type1 = remainder // 100000
        remainder = remainder % 100000
        type2 = remainder // 10000
        remainder = remainder % 10000
        type3 = remainder // 1000
        remainder = remainder % 1000
        type4 = remainder // 100
        remainder = remainder % 100
        type5 = remainder // 10
        type6 = remainder % 10
        return [type1, type2, type3, type4, type5, type6]


def encode_mask_value(types: list) -> int:
    """Encode list of types to mask value.
    
    Args:
        types: List of type integers (1-6)
        
    Returns:
        Encoded mask value
    """
    if len(types) == 0:
        return 0
    elif len(types) == 1:
        return types[0]
    elif len(types) == 2:
        sorted_types = sorted(types)
        return 10 + sorted_types[0] * 10 + sorted_types[1]
    elif len(types) == 3:
        sorted_types = sorted(types)
        return 100 + sorted_types[0] * 100 + sorted_types[1] * 10 + sorted_types[2]
    elif len(types) == 4:
        sorted_types = sorted(types)
        return 1000 + sorted_types[0] * 1000 + sorted_types[1] * 100 + sorted_types[2] * 10 + sorted_types[3]
    elif len(types) == 5:
        sorted_types = sorted(types)
        return 10000 + sorted_types[0] * 10000 + sorted_types[1] * 1000 + sorted_types[2] * 100 + sorted_types[3] * 10 + sorted_types[4]
    else:  # 6 types
        sorted_types = sorted(types)
        return 100000 + sorted_types[0] * 100000 + sorted_types[1] * 10000 + sorted_types[2] * 1000 + sorted_types[3] * 100 + sorted_types[4] * 10 + sorted_types[5]


def create_combined_head_matrix(
    previous_scores: torch.Tensor,
    duplicate_scores: torch.Tensor,
    induction_scores: torch.Tensor,
    truthfulness_scores: torch.Tensor = None,
    truthfulness_heads_list: list = None,
    retrieval_heads_list: list = None,
    iteration_scores: torch.Tensor = None,
    top_k: int = 30
) -> tuple:
    """Create a combined matrix showing which heads belong to which types.
    
    Uses vertical splits for overlapping heads (keeps white for multi-type cells,
    will be drawn as split rectangles in visualization).
    
    Args:
        previous_scores: Tensor of shape (n_layers, n_heads) for previous token heads
        duplicate_scores: Tensor of shape (n_layers, n_heads) for duplicate token heads
        induction_scores: Tensor of shape (n_layers, n_heads) for induction heads
        truthfulness_scores: Optional tensor of shape (n_layers, n_heads) for truthfulness heads
        truthfulness_heads_list: Optional list of (layer, head) tuples for truthfulness heads
        retrieval_heads_list: Optional list of (layer, head) tuples for retrieval heads
        top_k: Number of top heads to consider for each type
        
    Returns:
        Tuple of (color_matrix, mask_matrix)
        - color_matrix: (n_layers, n_heads, 3) RGB matrix for visualization
        - mask_matrix: (n_layers, n_heads) matrix with type information (0=white, 1=previous, 2=duplicate, 3=induction, 4=truthfulness, 5=retrieval)
    """
    n_layers, n_heads = previous_scores.shape
    
    # Get top k heads for each type
    _, _, prev_layers, prev_heads = get_top_k_heads(previous_scores, k=top_k)
    _, _, dup_layers, dup_heads = get_top_k_heads(duplicate_scores, k=top_k)
    _, _, ind_layers, ind_heads = get_top_k_heads(induction_scores, k=top_k)
    
    # Create sets of top heads for each type
    # Convert tensors to lists of integers
    prev_set = set((int(l.item()), int(h.item())) for l, h in zip(prev_layers, prev_heads))
    dup_set = set((int(l.item()), int(h.item())) for l, h in zip(dup_layers, dup_heads))
    ind_set = set((int(l.item()), int(h.item())) for l, h in zip(ind_layers, ind_heads))
    
    # Get truthfulness heads set
    truth_set = set()
    if truthfulness_heads_list is not None:
        truth_set = set((int(layer), int(head)) for layer, head in truthfulness_heads_list)
    elif truthfulness_scores is not None:
        # Get top k truthfulness heads from scores
        _, _, truth_layers, truth_heads = get_top_k_heads(truthfulness_scores, k=top_k)
        truth_set = set((int(l.item()), int(h.item())) for l, h in zip(truth_layers, truth_heads))
    
    # Get retrieval heads set
    retrieval_set = set()
    if retrieval_heads_list is not None:
        for item in retrieval_heads_list:
            # Handle different formats: (layer, head), (layer, head, score), or [layer, head]
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    # Extract layer and head (ignore score if present)
                    layer, head = item[0], item[1]
                    # Handle case where layer or head might be a list
                    if isinstance(layer, list):
                        layer = layer[0] if len(layer) > 0 else layer
                    if isinstance(head, list):
                        head = head[0] if len(head) > 0 else head
                    retrieval_set.add((int(layer), int(head)))
    
    # Get iteration heads set
    iteration_set = set()
    if iteration_scores is not None:
        # Get top k iteration heads from scores
        _, _, iter_layers, iter_heads = get_top_k_heads(iteration_scores, k=top_k)
        iteration_set = set((int(l.item()), int(h.item())) for l, h in zip(iter_layers, iter_heads))
    
    # Debug: print set sizes
    print(f"  Previous heads: {len(prev_set)}")
    print(f"  Duplicate heads: {len(dup_set)}")
    print(f"  Induction heads: {len(ind_set)}")
    print(f"  Truthfulness heads: {len(truth_set)}")
    print(f"  Retrieval heads: {len(retrieval_set)}")
    print(f"  Iteration heads: {len(iteration_set)}")
    print(f"  Overlap (prev & dup): {len(prev_set & dup_set)}")
    print(f"  Overlap (prev & ind): {len(prev_set & ind_set)}")
    print(f"  Overlap (prev & truth): {len(prev_set & truth_set)}")
    print(f"  Overlap (prev & retrieval): {len(prev_set & retrieval_set)}")
    print(f"  Overlap (prev & iteration): {len(prev_set & iteration_set)}")
    print(f"  Overlap (dup & ind): {len(dup_set & ind_set)}")
    print(f"  Overlap (dup & truth): {len(dup_set & truth_set)}")
    print(f"  Overlap (dup & retrieval): {len(dup_set & retrieval_set)}")
    print(f"  Overlap (dup & iteration): {len(dup_set & iteration_set)}")
    print(f"  Overlap (ind & truth): {len(ind_set & truth_set)}")
    print(f"  Overlap (ind & retrieval): {len(ind_set & retrieval_set)}")
    print(f"  Overlap (ind & iteration): {len(ind_set & iteration_set)}")
    print(f"  Overlap (truth & retrieval): {len(truth_set & retrieval_set)}")
    print(f"  Overlap (truth & iteration): {len(truth_set & iteration_set)}")
    print(f"  Overlap (retrieval & iteration): {len(retrieval_set & iteration_set)}")
    print(f"  Overlap (all six): {len(prev_set & dup_set & ind_set & truth_set & retrieval_set & iteration_set)}")
    
    # Define colors for each type (RGB) - Muted academic color scheme
    # Using softer, less saturated colors for better academic appearance
    # Previous Head: Muted Blue
    color_previous = np.array([70, 130, 180])  # Steel blue (muted)
    # Duplicate Head: Muted Green
    color_duplicate = np.array([85, 139, 85])  # Sage green (muted)
    # Induction Head: Muted Orange
    color_induction = np.array([205, 133, 63])  # Peru/tan (muted)
    # Truthfulness Head: Muted Pink/Rose
    color_truthfulness = np.array([188, 143, 143])  # Rosy brown (muted pink)
    # Retrieval Head: Muted Purple
    color_retrieval = np.array([128, 100, 162])  # Muted purple
    # Iteration Head: Muted Brown
    color_iteration = np.array([139, 115, 85])  # Muted brown
    
    # Create color matrix
    color_matrix = np.ones((n_layers, n_heads, 3)) * 255  # White background
    mask_matrix = np.zeros((n_layers, n_heads))  # 0 = white, 1=prev, 2=dup, 3=ind, etc.
    
    # Fill in colors for each head
    for layer in range(n_layers):
        for head in range(n_heads):
            types = []
            if (layer, head) in prev_set:
                types.append(1)  # Previous
            if (layer, head) in dup_set:
                types.append(2)  # Duplicate
            if (layer, head) in ind_set:
                types.append(3)  # Induction
            if (layer, head) in truth_set:
                types.append(4)  # Truthfulness
            if (layer, head) in retrieval_set:
                types.append(5)  # Retrieval
            if (layer, head) in iteration_set:
                types.append(6)  # Iteration
            
            if len(types) == 0:
                # No type, keep white
                color_matrix[layer, head] = [255, 255, 255]
                mask_matrix[layer, head] = 0
            elif len(types) == 1:
                # Single type - fill with that color
                if 1 in types:
                    color_matrix[layer, head] = color_previous
                    mask_matrix[layer, head] = 1
                elif 2 in types:
                    color_matrix[layer, head] = color_duplicate
                    mask_matrix[layer, head] = 2
                elif 3 in types:
                    color_matrix[layer, head] = color_induction
                    mask_matrix[layer, head] = 3
                elif 4 in types:
                    color_matrix[layer, head] = color_truthfulness
                    mask_matrix[layer, head] = 4
                elif 5 in types:
                    color_matrix[layer, head] = color_retrieval
                    mask_matrix[layer, head] = 5
                elif 6 in types:
                    color_matrix[layer, head] = color_iteration
                    mask_matrix[layer, head] = 6
            else:
                # Multiple types - keep white, encode types in mask for vertical splitting
                color_matrix[layer, head] = [255, 255, 255]
                mask_matrix[layer, head] = encode_mask_value(types)
    
    return color_matrix, mask_matrix


def visualize_combined_heads(
    previous_scores: torch.Tensor,
    duplicate_scores: torch.Tensor,
    induction_scores: torch.Tensor,
    truthfulness_scores: torch.Tensor = None,
    truthfulness_heads_list: list = None,
    retrieval_heads_list: list = None,
    iteration_scores: torch.Tensor = None,
    top_k: int = 30,
    save_path: Path = None,
    model_name: str = None
):
    """Visualize all head types in one combined matrix with vertical splits.
    
    Args:
        previous_scores: Tensor of shape (n_layers, n_heads) for previous token heads
        duplicate_scores: Tensor of shape (n_layers, n_heads) for duplicate token heads
        induction_scores: Tensor of shape (n_layers, n_heads) for induction heads
        truthfulness_scores: Optional tensor of shape (n_layers, n_heads) for truthfulness heads
        truthfulness_heads_list: Optional list of (layer, head) tuples for truthfulness heads
        retrieval_heads_list: Optional list of (layer, head) tuples for retrieval heads
        top_k: Number of top heads to consider for each type
        save_path: Optional path to save the figure
    """
    # Create combined matrix
    color_matrix, mask_matrix = create_combined_head_matrix(
        previous_scores, duplicate_scores, induction_scores, 
        truthfulness_scores, truthfulness_heads_list, retrieval_heads_list, 
        iteration_scores, top_k
    )
    
    n_layers, n_heads = previous_scores.shape
    
    # Debug: print some statistics
    print(f"  Color matrix shape: {color_matrix.shape}")
    print(f"  Color matrix range: [{color_matrix.min()}, {color_matrix.max()}]")
    print(f"  Non-white cells: {(mask_matrix > 0).sum()} / {n_layers * n_heads}")
    
    # Transpose for visualization: columns (x-axis) = layers, rows (y-axis) = heads
    # For imshow, we need (height, width, channels) = (n_heads, n_layers, 3)
    color_matrix_viz = color_matrix.transpose(1, 0, 2)  # (n_heads, n_layers, 3)
    mask_matrix_viz = mask_matrix.T  # (n_heads, n_layers)
    
    # Prefer matplotlib as it's more reliable for RGB images with vertical splits
    if HAS_MATPLOTLIB:
        print("  Using matplotlib for visualization...")
        has_truthfulness = (truthfulness_scores is not None) or (truthfulness_heads_list is not None)
        has_retrieval = retrieval_heads_list is not None
        has_iteration = iteration_scores is not None
        visualize_combined_with_matplotlib(
            color_matrix_viz, mask_matrix_viz, n_layers, n_heads, top_k, save_path,
            has_truthfulness=has_truthfulness, has_retrieval=has_retrieval, has_iteration=has_iteration,
            model_name=model_name
        )
    elif HAS_PLOTLY:
        print("  Using plotly for visualization...")
        visualize_combined_with_plotly(color_matrix_viz, mask_matrix_viz, n_layers, n_heads, top_k, save_path, model_name=model_name)
    else:
        print("  Cannot visualize: no plotting library available")


def visualize_combined_with_plotly(
    color_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    n_layers: int,
    n_heads: int,
    top_k: int,
    save_path: Path = None,
    model_name: str = None
):
    """Visualize combined heads using Plotly.
    
    Note: Plotly doesn't support vertical splits well, so this is a fallback.
    For best results, use matplotlib.
    """
    print(f"  Plotly - Color matrix shape: {color_matrix.shape}")
    print(f"  Plotly - Color matrix range: [{color_matrix.min()}, {color_matrix.max()}]")
    print("  Warning: Plotly visualization doesn't support vertical splits well.")
    print("  Consider using matplotlib for better visualization with vertical splits.")
    
    # Use PIL Image method
    try:
        from PIL import Image
        import io
        import base64
        
        print("  Converting to PIL Image...")
        # Convert to PIL Image (ensure uint8)
        color_matrix_uint8 = np.clip(color_matrix.astype(np.uint8), 0, 255)
        img = Image.fromarray(color_matrix_uint8, 'RGB')
        
        print("  Converting to base64...")
        # Convert to base64 for embedding
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        img_data = f"data:image/png;base64,{img_str}"
        
        print("  Creating Plotly figure...")
        # Create figure
        fig = go.Figure()
        
        # Add image to plot
        fig.add_layout_image(
            dict(
                source=img_data,
                xref="x",
                yref="y",
                x=-0.5,
                y=-0.5,  # Bottom-left corner
                sizex=n_layers,
                sizey=n_heads,
                sizing="stretch",
                layer="below"
            )
        )
        
        print("  Adding grid lines...")
        # Add grid lines
        for i in range(n_layers + 1):
            fig.add_shape(
                type="line",
                x0=i - 0.5, x1=i - 0.5,
                y0=-0.5, y1=n_heads - 0.5,
                line=dict(color="black", width=2),
                layer="above"
            )
        
        for i in range(n_heads + 1):
            fig.add_shape(
                type="line",
                x0=-0.5, x1=n_layers - 0.5,
                y0=i - 0.5, y1=i - 0.5,
                line=dict(color="black", width=2),
                layer="above"
            )
        
    except ImportError:
        print("  PIL not available, using fallback method (slower)...")
        # Fallback: only add shapes for non-white cells
        fig = go.Figure()
        
        non_white_count = 0
        for head_idx in range(n_heads):
            for layer_idx in range(n_layers):
                color = color_matrix[head_idx, layer_idx]
                # Only add shape if not white
                if not (color[0] == 255 and color[1] == 255 and color[2] == 255):
                    color_rgb = f"rgb({int(color[0])}, {int(color[1])}, {int(color[2])})"
                    fig.add_shape(
                        type="rect",
                        x0=layer_idx - 0.5,
                        x1=layer_idx + 0.5,
                        y0=head_idx - 0.5,
                        y1=head_idx + 0.5,
                        fillcolor=color_rgb,
                        line=dict(color="black", width=2),
                        layer="below"
                    )
                    non_white_count += 1
        
        print(f"  Added {non_white_count} non-white cells")
        
        # Add grid lines
        for i in range(n_layers + 1):
            fig.add_shape(
                type="line",
                x0=i - 0.5, x1=i - 0.5,
                y0=-0.5, y1=n_heads - 0.5,
                line=dict(color="black", width=2),
                layer="above"
            )
        
        for i in range(n_heads + 1):
            fig.add_shape(
                type="line",
                x0=-0.5, x1=n_layers - 0.5,
                y0=i - 0.5, y1=i - 0.5,
                line=dict(color="black", width=2),
                layer="above"
            )
    
    # Update layout
    # Set title with model name
    if model_name:
        title_text = f"{model_name} - Combined Head Types (Top {top_k})"
    else:
        title_text = f"Combined Head Types (Top {top_k})"
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=22, family="Arial, sans-serif", color='black'),
            x=0.5,
            xanchor='center',
            pad=dict(t=10, b=20)
        ),
        xaxis=dict(
            title=dict(text="Layer Index", font=dict(size=18, family="Arial, sans-serif")),
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[-0.5, n_layers - 0.5],
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            tickfont=dict(size=14, family="Arial, sans-serif")
        ),
        yaxis=dict(
            title=dict(text="Head Index", font=dict(size=18, family="Arial, sans-serif")),
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[n_heads - 0.5, -0.5],
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            autorange='reversed',
            tickfont=dict(size=14, family="Arial, sans-serif")
        ),
        width=1000,
        height=800,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Save or show
    print("  Updating layout...")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Saving to {save_path}...")
        try:
            fig.write_image(str(save_path.with_suffix('.png')), width=1400, height=1200, scale=2)
            print(f"  ✓ Saved PNG: {save_path.with_suffix('.png')}")
        except Exception as e:
            print(f"  PNG export failed: {e}, trying HTML...")
            fig.write_html(str(save_path.with_suffix('.html')))
            print(f"  ✓ Saved HTML: {save_path.with_suffix('.html')}")
    else:
        print("  Showing figure...")
        fig.show()


def visualize_combined_with_matplotlib(
    color_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    n_layers: int,
    n_heads: int,
    top_k: int,
    save_path: Path = None,
    has_truthfulness: bool = False,
    has_retrieval: bool = False,
    has_iteration: bool = False,
    model_name: str = None
):
    """Visualize combined heads using Matplotlib with vertical splits.
    
    This function draws vertical splits for multi-type cells, clearly showing
    which heads have overlapping functions.
    """
    import matplotlib.pyplot as plt
    
    # Debug: check color matrix
    print(f"  Matplotlib - Color matrix shape: {color_matrix.shape}")
    print(f"  Matplotlib - Color matrix dtype: {color_matrix.dtype}")
    print(f"  Matplotlib - Color matrix range: [{color_matrix.min()}, {color_matrix.max()}]")
    
    # Set matplotlib styling with larger fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    # Force matplotlib to show all ticks
    plt.rcParams['xtick.major.pad'] = 2
    plt.rcParams['ytick.major.pad'] = 2
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Ensure color matrix is uint8 and in correct range
    color_matrix_uint8 = np.clip(color_matrix.astype(np.uint8), 0, 255)
    
    # Display RGB image for single-type cells
    # imshow expects (height, width, channels) = (n_heads, n_layers, 3)
    print(f"  Displaying image with shape {color_matrix_uint8.shape}")
    
    ax.imshow(color_matrix_uint8, aspect='auto', origin='upper', 
              extent=[-0.5, n_layers - 0.5, n_heads - 0.5, -0.5], 
              interpolation='nearest', filternorm=False)
    
    # Set y-axis limits - we'll invert it so head 0 is at top visually
    ax.set_ylim(n_heads - 0.5, -0.5)  # Inverted: top is -0.5, bottom is n_heads-0.5
    
    # Draw vertical splits for multi-type cells
    from matplotlib.patches import Rectangle
    color_map = {
        1: np.array([70, 130, 180]) / 255.0,   # Previous (steel blue, muted)
        2: np.array([85, 139, 85]) / 255.0,    # Duplicate (sage green, muted)
        3: np.array([205, 133, 63]) / 255.0,    # Induction (peru/tan, muted)
        4: np.array([188, 143, 143]) / 255.0,    # Truthfulness (rosy brown, muted)
        5: np.array([128, 100, 162]) / 255.0,    # Retrieval (muted purple)
        6: np.array([139, 115, 85]) / 255.0     # Iteration (muted brown)
    }
    
    print("  Drawing vertical splits for multi-type cells...")
    print(f"  Mask matrix shape: {mask_matrix.shape}")
    multi_type_count = 0
    for head_idx in range(n_heads):
        for layer_idx in range(n_layers):
            # mask_matrix is (n_heads, n_layers) after transpose
            mask_value = mask_matrix[head_idx, layer_idx]
            
            # Decode mask to get types
            types = decode_mask_value(mask_value)
            
            if len(types) <= 1:
                continue  # Single type or white, already handled by imshow
            
            multi_type_count += 1
            
            # Draw vertical splits
            num_types = len(types)
            cell_width = 1.0  # Full cell width
            segment_width = cell_width / num_types
            
            x_left = layer_idx - 0.5
            y_bottom = head_idx - 0.5
            
            for i, head_type in enumerate(types):
                x_start = x_left + i * segment_width
                color = color_map[head_type]
                
                rect = Rectangle(
                    (x_start, y_bottom),
                    segment_width,
                    1.0,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2,
                    zorder=5
                )
                ax.add_patch(rect)
    
    print(f"  Drew {multi_type_count} multi-type cells with vertical splits")
    
    # Add grid lines
    for i in range(n_layers + 1):
        ax.axvline(x=i - 0.5, color='black', linewidth=2, alpha=1.0, zorder=10)
    
    for i in range(n_heads + 1):
        ax.axhline(y=i - 0.5, color='black', linewidth=2, alpha=1.0, zorder=10)
    
    # Set labels with larger fonts
    ax.set_xlabel('Layer Index', fontsize=18, labelpad=10)
    ax.set_ylabel('Head Index', fontsize=18, labelpad=10)
    # Set title with model name
    if model_name:
        title = f"{model_name} - Combined Head Types (Top {top_k})"
    else:
        title = f"Combined Head Types (Top {top_k})"
    ax.set_title(title, fontsize=22, pad=20)
    
    # Set ticks - only show 0 and max (31) for both axes
    ax.set_xticks([0, n_layers - 1])
    ax.set_yticks([0, n_heads - 1])
    ax.set_xticklabels([0, n_layers - 1])
    ax.set_yticklabels([0, n_heads - 1])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='x', which='major', length=5, width=1)
    ax.tick_params(axis='y', which='major', length=5, width=1)
    ax.set_xlim(-0.5, n_layers - 0.5)
    # ylim is already set above in imshow extent
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add legend with clearer labels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[70/255, 130/255, 180/255], label='Previous Token Head'),
        Patch(facecolor=[85/255, 139/255, 85/255], label='Duplicate Token Head'),
        Patch(facecolor=[205/255, 133/255, 63/255], label='Induction Head'),
    ]
    # Add truthfulness head to legend if present
    if has_truthfulness:
        legend_elements.append(Patch(facecolor=[188/255, 143/255, 143/255], label='Truthfulness Head'))
    # Add retrieval head to legend if present
    if has_retrieval:
        legend_elements.append(Patch(facecolor=[128/255, 100/255, 162/255], label='Retrieval Head'))
    # Add iteration head to legend if present
    if has_iteration:
        legend_elements.append(Patch(facecolor=[139/255, 115/255, 85/255], label='Iteration Head'))
    legend_elements.append(Patch(facecolor='white', edgecolor='black', linewidth=1, label='None'))
    
    # Create legend positioned at the bottom of the plot, below x-axis label
    # Use 4 columns to keep legend compact (2 rows if needed)
    ncol = 4
    
    legend = ax.legend(
        handles=legend_elements, 
        bbox_to_anchor=(0.5, -0.15),  # Position further below to avoid x-axis label
        loc='upper center',  # Anchor point for bbox_to_anchor
        fontsize=14,  # Larger font
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        title='Head Type ',
        title_fontsize=16,  # Larger title
        edgecolor='black',
        borderpad=0.8,  # Reduce padding
        ncol=ncol,  # Use 4 columns (will wrap to 2 rows if needed)
        columnspacing=1.0  # Reduce spacing between columns
    )
    # Add note about overlapping at the very bottom
    fig.text(0.5, 0.005, 'Note: Cells with vertical color splits indicate heads with overlapping functions', 
             ha='center', va='bottom', fontsize=14, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8, edgecolor='black'))
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Adjust layout to make room for the legend and x-axis label at the bottom
    plt.tight_layout(rect=[0, 0.18, 1, 1])  # Leave more space at the bottom for legend and x-axis label
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', format='png')
        print(f"  Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def visualize_head_scores(scores: torch.Tensor, title: str, top_k: int = 30, save_path: Path = None):
    """Visualize head scores with top k heads highlighted.
    
    Args:
        scores: Tensor of shape (n_layers, n_heads)
        title: Plot title
        top_k: Number of top heads to highlight
        save_path: Optional path to save the figure
    """
    print(f"\n{title}")
    print(f"  Shape: {scores.shape}")
    print(f"  Score range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")
    
    # Get top k heads info
    top_values, _, top_layers, top_heads = get_top_k_heads(scores, k=top_k)
    print(f"  Top {top_k} heads:")
    for i, (layer, head, score) in enumerate(zip(top_layers[:10], top_heads[:10], top_values[:10]), 1):
        print(f"    {i}. Layer {layer.item()}, Head {head.item()}: {score.item():.3f}")
    if top_k > 10:
        print(f"    ... ({top_k - 10} more)")
    
    # Visualize
    if HAS_PLOTLY:
        visualize_with_neel_plotly(scores, title, top_k, save_path)
    elif HAS_MATPLOTLIB:
        visualize_with_matplotlib(scores, title, top_k, save_path)
    else:
        print("  Cannot visualize: no plotting library available")
        print("  Please install plotly (pip install plotly kaleido) or matplotlib (pip install matplotlib)")


def main():
    """Main function to load and visualize head scores."""
    # Define paths
    # model_name = "Llama-2-7b-hf"

    model_name ="Meta-Llama-3-8B-Instruct"

    truthfulness_json_path = Path("/home/jhe/Head_analysis/Retrieval_Head/head_score_all/truthfulness_head/truthfulness_head_Meta-Llama-3-8B-Instruct_avg.npy")

    iteration_file = Path("/home/jhe/Head_analysis/Retrieval_Head/head_score_all/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_iteration_heads_inv_gt_0.70_sorted.npy")

    base_dir = Path("/home/jhe/Head_analysis/Retrieval_Head/head_score_all")
    output_dir = Path("/home/jhe/Head_analysis/Retrieval_Head/head_score_all/viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Retrieval heads list (convert from list of lists to list of tuples)
    # retrieval_head_list = [[11, 15], [16, 19], [8, 26], [6, 16], [7, 12], [17, 22], [7, 10], [19, 15], [6, 9], [21, 30], [24, 29], [11, 2], [6, 30], [18, 30], [21, 1], [15, 14], [10, 18], [17, 18], [22, 22], [17, 0], [14, 7], [20, 30], [19, 10], [14, 24], [20, 0], [7, 4], [21, 4], [21, 16], [20, 29], [22, 19], [24, 30], [11, 14], [20, 10], [13, 23], [22, 8], [29, 19], [20, 28], [23, 31], [18, 10], [11, 30], [23, 20], [8, 22], [24, 3], [14, 29], [16, 1], [14, 18], [8, 31], [12, 16], [21, 28], [20, 1]]
    # retrieval_heads_list = [(layer, head) for layer, head in retrieval_head_list]  # Convert to tuples
    
    # Path to truthfulness heads JSON file

    retrieval_path = base_dir / model_name / f"{model_name}_retrieval_head.json"
    with open(retrieval_path) as file:
        head_list = json.loads(file.readline())
    ## use the average retrieval score and ranking
    head_score_list = [([int(ll) for ll in l[0].split("-")],np.mean(l[1])) for l in head_list.items()]
    head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True) 
    top_retrieval_heads = [[l[0],  round(np.mean(l[1]), 2)] for l in head_score_list][:10]

    retrieval_heads_list = [(layer, head) for layer, head in top_retrieval_heads]


    # truthfulness_json_path = base_dir / model_name / f"{model_name}_truthfulness_head.json"

    
    # Load the three main head types for combined visualization (Custom Model)
    previous_file = base_dir / base_dir / model_name / f"{model_name}_previous_token_head_custom_abs.pt"
    duplicate_file = base_dir / model_name / f"{model_name}_duplicate_token_head_custom_abs.pt"
    induction_file = base_dir / model_name / f"{model_name}_induction_head_custom_abs.pt"
    
    print("="*60)
    print("Loading head scores for combined visualization (Custom Model)...")
    print("="*60)
    
    # Load the three types
    previous_scores = None
    duplicate_scores = None
    induction_scores = None
    
    if previous_file.exists():
        previous_scores = load_head_scores(previous_file)
        print(f"✓ Loaded Previous Token Head (ABS): shape {previous_scores.shape}")
    else:
        print(f"✗ File not found: {previous_file}")
    
    if duplicate_file.exists():
        duplicate_scores = load_head_scores(duplicate_file)
        print(f"✓ Loaded Duplicate Token Head (ABS): shape {duplicate_scores.shape}")
    else:
        print(f"✗ File not found: {duplicate_file}")
    
    if induction_file.exists():
        induction_scores = load_head_scores(induction_file)
        print(f"✓ Loaded Induction Head (ABS): shape {induction_scores.shape}")
    else:
        print(f"✗ File not found: {induction_file}")
    
    # Check if all three are loaded
    if previous_scores is None or duplicate_scores is None or induction_scores is None:
        print("Error: Need all three head score files for combined visualization!")
        return
    
    # Verify shapes match
    if not (previous_scores.shape == duplicate_scores.shape == induction_scores.shape):
        print("Error: Head score shapes don't match!")
        print(f"  Previous: {previous_scores.shape}")
        print(f"  Duplicate: {duplicate_scores.shape}")
        print(f"  Induction: {induction_scores.shape}")
        return
    
    # Load truthfulness heads
    truthfulness_scores = None
    truthfulness_heads_list = None
    if truthfulness_json_path.exists():
        print("\n" + "="*60)
        print("Loading Truthfulness Heads...")
        print("="*60)
        n_layers, n_heads = previous_scores.shape
        truthfulness_scores, truthfulness_heads_list = load_truthfulness_heads(
            truthfulness_json_path, n_layers=n_layers, n_heads=n_heads
        )
        print(f"✓ Loaded Truthfulness Heads: {len(truthfulness_heads_list)} heads")
    else:
        print(f"\n⚠ Truthfulness heads file not found: {truthfulness_json_path}")
        print("  Continuing without truthfulness heads...")
    
    # Load iteration heads (Custom Model)
    iteration_scores = None
    # iteration_file = base_dir / "iteration_heads_inv_gt_0.70_sorted——custom.npy"
    if iteration_file.exists():
        print("\n" + "="*60)
        print("Loading Iteration Heads...")
        print("="*60)
        n_layers, n_heads = previous_scores.shape
        iteration_scores = load_iteration_heads(iteration_file, n_layers=n_layers, n_heads=n_heads)
        print(f"✓ Loaded Iteration Heads")
    else:
        print(f"\n⚠ Iteration heads file not found: {iteration_file}")
        print("  Continuing without iteration heads...")
    
    # Create combined visualization
    print("\n" + "="*60)
    print("Creating combined visualization with vertical splits...")
    print("="*60)
    
    save_path = output_dir / f"{model_name}_combined_head_types_all_vertical_split_top30_custom.png"
    visualize_combined_heads(
        previous_scores,
        duplicate_scores,
        induction_scores,
        truthfulness_scores=truthfulness_scores,
        truthfulness_heads_list=truthfulness_heads_list,
        retrieval_heads_list=retrieval_heads_list,
        iteration_scores=iteration_scores,
        top_k=30,
        save_path=save_path,
        model_name=model_name
    )
    
    print("\n" + "="*60)
    print("Combined visualization completed!")
    print(f"Saved to: {save_path}")
    print("="*60)


if __name__ == "__main__":
    main()

