"""
Custom Model Head Detector (LLaMA2-7B Version)

A tool for automatically detecting specialized attention heads in transformer models
using custom LlamaForCausalLM model (via CustomModelAdapter).

This version uses CustomModelAdapter instead of TransformerLens's HookedTransformer,
allowing you to use custom model implementations while maintaining the same API.

Supports detection of:
- Previous token heads
- Duplicate token heads  
- Induction heads
- Custom detection patterns

This version is configured for LLaMA2-7B model. To use other models:
- Change model name to any HuggingFace model compatible with LlamaForCausalLM

Note: LLaMA models require significant GPU memory (recommended: 16GB+ VRAM).
If you encounter out-of-memory errors:
1. Use torch_dtype=torch.float16 to reduce memory usage
2. Use shorter sequences
3. Consider using a smaller model variant

How previous_token_head detection works:
---------------------------------------
1. Generate detection pattern:
   - For a sequence of length n, create an n×n matrix
   - Set pattern[i, i-1] = 1 for all i > 0 (sub-diagonal)
   - This represents: "each token should attend to its previous token"
   
2. Get actual attention patterns:
   - Run model forward pass to get attention weights
   - Extract attention pattern for each head: shape (seq_len, seq_len)
   
3. Compute similarity:
   - Compare actual pattern with detection pattern
   - Two methods:
     * "mul": score = sum(actual * detection) / sum(actual)
       → What fraction of attention matches the pattern?
     * "abs": score = 1 - mean_absolute_difference
       → How close is the pattern? (recommended)
   
4. Return scores:
   - Matrix of shape (n_layers, n_heads)
   - Each value is similarity score [-1, 1]
   - Higher score = better match to previous_token_head pattern

Example:
    >>> from model_adapter import CustomModelAdapter
    >>> from head_detect_llama_custom import detect_head, plot_head_detection_scores
    >>> import torch
    >>> 
    >>> # Load custom model
    >>> model = CustomModelAdapter.from_pretrained(
    ...     "meta-llama/Llama-2-7b-hf", 
    ...     device="cuda", 
    ...     torch_dtype=torch.bfloat16,
    ...     use_flash_attention_2=False  # Disable Flash Attention 2 for accurate attention weights
    ... )
    >>> 
    >>> scores = detect_head(model, "Your prompt here", "previous_token_head")
    >>> plot_head_detection_scores(scores)
    
    # Find the best matching head
    >>> layer, head = scores.argmax().div(scores.shape[1], rounding_mode='floor'), scores.argmax() % scores.shape[1]
    >>> print(f"Best previous token head: Layer {layer}, Head {head}, Score: {scores[layer, head]:.3f}")
"""

from collections import defaultdict
import logging
import os
from pathlib import Path
from typing import cast, Dict, List, Optional, Tuple, Union
from typing_extensions import get_args, Literal
import sys

import numpy as np
import torch
import einops
from tqdm import tqdm

# Add path to import CustomModelAdapter
_current_dir = Path(__file__).parent
_retrieval_head_dir = _current_dir.parent.parent / "Retrieval_Head"
sys.path.insert(0, str(_current_dir))
sys.path.insert(0, str(_retrieval_head_dir / "faiss_attn"))
sys.path.insert(0, str(_retrieval_head_dir))

try:
    from model_adapter import CustomModelAdapter
    from hook_utils import ActivationCache
    HAS_CUSTOM_MODEL = True
except ImportError as e:
    HAS_CUSTOM_MODEL = False
    print(f"⚠️  Warning: Could not import CustomModelAdapter: {e}")
    print(f"   Current directory: {_current_dir}")
    print(f"   Retrieval_Head directory: {_retrieval_head_dir}")
    print("   Please ensure model_adapter.py and hook_utils.py are available.")
    CustomModelAdapter = None
    ActivationCache = None

# ============================================================================
# Optional imports for visualization
# ============================================================================
# These are optional dependencies - core functionality works without them
# pysvelte: 用于可视化注意力模式（需要 Node.js）
# neel_plotly: Neel Nanda 的 Plotly 工具集，用于绘制热力图

try:
    import pysvelte
    HAS_PYSVELTE = True
except ImportError:
    HAS_PYSVELTE = False
    pysvelte = None

try:
    from neel_plotly import line, imshow, scatter
    HAS_NEEL_PLOTLY = True
except ImportError:
    HAS_NEEL_PLOTLY = False
    # Create dummy functions if neel_plotly is not available
    def imshow(*args, **kwargs):
        raise ImportError(
            "neel_plotly is not installed. Install it with:\n"
            "  pip install git+https://github.com/neelnanda-io/neel-plotly.git\n"
            "Or use --user flag: pip install --user git+https://github.com/neelnanda-io/neel-plotly.git"
        )
    def line(*args, **kwargs):
        raise ImportError("neel_plotly is not installed. See imshow() error message for installation instructions.")
    def scatter(*args, **kwargs):
        raise ImportError("neel_plotly is not installed. See imshow() error message for installation instructions.")


# ============================================================================
# Type definitions
# ============================================================================

HeadName = Literal["previous_token_head", "duplicate_token_head", "induction_head"]
HEAD_NAMES = cast(List[HeadName], get_args(HeadName))
ErrorMeasure = Literal["abs", "mul"]

LayerHeadTuple = Tuple[int, int]
LayerToHead = Dict[int, List[int]]


# ============================================================================
# Error messages
# ============================================================================

INVALID_HEAD_NAME_ERR = (
    f"detection_pattern must be a Tensor or one of head names: {HEAD_NAMES}; got %s"
)

SEQ_LEN_ERR = (
    "The sequence must be non-empty and must fit within the model's context window."
)

DET_PAT_NOT_SQUARE_ERR = (
    "The detection pattern must be a lower triangular matrix of shape "
    "(sequence_length, sequence_length); sequence_length=%d; got detection pattern of shape %s"
)


# ============================================================================
# Utility functions
# ============================================================================

def is_square(x: torch.Tensor) -> bool:
    """Checks if `x` is a square matrix."""
    return x.ndim == 2 and x.shape[0] == x.shape[1]


def is_lower_triangular(x: torch.Tensor) -> bool:
    """Checks if `x` is a lower triangular matrix."""
    if not is_square(x):
        return False
    return x.equal(x.tril())


# ============================================================================
# Detection pattern generators
# ============================================================================

def get_previous_token_head_detection_pattern(
    tokens: torch.Tensor,  # [batch (1) x pos]
) -> torch.Tensor:
    """Generate detection pattern for previous token heads.
    
    Previous token heads attend to the immediately previous token at each position.
    Pattern: diagonal of 1's below the main diagonal.
    
    Calculation process:
    --------------------
    1. Create a zero matrix of shape (seq_len, seq_len)
    2. Fill the sub-diagonal (one position below main diagonal) with 1's
    3. Return lower triangular matrix
    
    Example for sequence length 5:
        Input tokens: [BOS, "The", "cat", "sat", "on"]
        
        Detection pattern (5x5):
        [[0, 0, 0, 0, 0],   # Position 0 (BOS): no previous token
         [1, 0, 0, 0, 0],   # Position 1 ("The"): attend to position 0
         [0, 1, 0, 0, 0],   # Position 2 ("cat"): attend to position 1
         [0, 0, 1, 0, 0],   # Position 3 ("sat"): attend to position 2
         [0, 0, 0, 1, 0]]   # Position 4 ("on"): attend to position 3
    
    This pattern represents: each token should attend to its immediate previous token.
    
    Args:
        tokens: Tokens being fed to the model.
    
    Returns:
        Lower triangular detection pattern matrix of shape (seq_len, seq_len).
    """
    seq_len = tokens.shape[-1]
    detection_pattern = torch.zeros(seq_len, seq_len)
    # Adds a diagonal of 1's below the main diagonal.
    # detection_pattern[1:, :-1] selects rows 1 to end, columns 0 to end-1
    # torch.eye(seq_len - 1) creates identity matrix, which becomes the sub-diagonal
    detection_pattern[1:, :-1] = torch.eye(seq_len - 1)
    return torch.tril(detection_pattern)


def get_duplicate_token_head_detection_pattern(
    tokens: torch.Tensor,  # [batch (1) x pos]
) -> torch.Tensor:
    """Generate detection pattern for duplicate token heads.
    
    Duplicate token heads attend to previous occurrences of the same token.
    
    Args:
        tokens: Tokens being fed to the model.
    
    Returns:
        Lower triangular detection pattern matrix.
    """
    # Remove batch dimension if present: [batch (1) x pos] -> [pos]
    if tokens.dim() > 1:
        tokens = tokens.squeeze(0)
    
    seq_len = tokens.shape[-1]
    
    # Create [pos x pos] matrix where pattern[i, j] = 1 if tokens[i] == tokens[j]
    # Using broadcasting: tokens[i] == tokens[j] for all i, j
    tokens_i = tokens.unsqueeze(1)  # [pos, 1]
    tokens_j = tokens.unsqueeze(0)  # [1, pos]
    eq_mask = (tokens_i == tokens_j).int()  # [pos, pos]

    # Fill diagonal with 0 - current token is always a duplicate of itself. Ignore that.
    eq_mask = eq_mask.clone()
    eq_mask[range(seq_len), range(seq_len)] = 0
    
    # Return lower triangular matrix
    return torch.tril(eq_mask.float())


def get_induction_head_detection_pattern(
    tokens: torch.Tensor,  # [batch (1) x pos]
) -> torch.Tensor:
    """Generate detection pattern for induction heads.
    
    Induction heads attend to the token after a duplicate token (shifted duplicate pattern).
    
    Args:
        tokens: Tokens being fed to the model.
    
    Returns:
        Lower triangular detection pattern matrix.
    """
    duplicate_pattern = get_duplicate_token_head_detection_pattern(tokens)

    # Shift all items one to the right
    shifted_tensor = torch.roll(duplicate_pattern, shifts=1, dims=1)

    # Replace first column with 0's
    # we don't care about bos but shifting to the right moves the last column to the first,
    # and the last column might contain non-zero values.
    zeros_column = torch.zeros(duplicate_pattern.shape[0], 1)
    result_tensor = torch.cat((zeros_column, shifted_tensor[:, 1:]), dim=1)
    return torch.tril(result_tensor)


def get_supported_heads() -> List[str]:
    """Returns a list of supported head names.
    
    Returns:
        List of supported head type names.
    """
    return list(HEAD_NAMES)


# ============================================================================
# Core similarity computation
# ============================================================================

def compute_head_attention_similarity_score(
    attention_pattern: torch.Tensor,  # [q_pos k_pos]
    detection_pattern: torch.Tensor,  # [seq_len seq_len] (seq_len == q_pos == k_pos)
    *,
    exclude_bos: bool,
    exclude_current_token: bool,
    error_measure: ErrorMeasure,
) -> float:
    """Compute the similarity between `attention_pattern` and `detection_pattern`.

    This function compares the actual attention pattern of a head with the expected
    detection pattern (e.g., previous_token_head pattern) and returns a similarity score.
    
    Calculation methods:
    -------------------
    
    1. "mul" method (element-wise multiplication):
       - Formula: score = sum(attention * detection) / sum(attention)
       - Interpretation: What fraction of attention is allocated to the expected positions?
       - Example: If detection pattern has 1 at (i, i-1) and attention has 0.8 there,
                   and total attention is 1.0, then score = 0.8/1.0 = 0.8
       - Range: [0, 1] where 1 means perfect match
    
    2. "abs" method (absolute difference, recommended):
       - Formula: score = 1 - mean_abs_diff
       - Interpretation: How close is the attention pattern to the expected pattern?
       - Example: If attention pattern exactly matches detection pattern:
                   abs_diff = 0, mean = 0, score = 1 - 0 = 1 (perfect match)
       - Range: [-1, 1] where 1 means perfect match, -1 means complete mismatch
       - Note: mean_abs_diff ranges from 0 to 2 (max difference between patterns)
    
    Args:
        attention_pattern: Lower triangular matrix representing the attention pattern 
                          of a particular attention head. Shape: (seq_len, seq_len)
        detection_pattern: Lower triangular matrix representing the attention pattern 
                          we are looking for. Shape: (seq_len, seq_len)
        exclude_bos: `True` if the beginning-of-sentence (BOS) token should be 
                    omitted from comparison.
        exclude_current_token: `True` if the current token at each position should be 
                              omitted from comparison.
        error_measure: "abs" for absolute difference, "mul" for element-wise multiplication.

    Returns:
        Similarity score (higher is better, typically in range [-1, 1]).
    """
    assert is_square(
        attention_pattern
    ), f"Attention pattern is not square; got shape {attention_pattern.shape}"

    # mul: element-wise multiplication (legacy method)
    if error_measure == "mul":
        # Create a copy to avoid modifying original
        attn = attention_pattern.clone()
        if exclude_bos:
            attn[:, 0] = 0  # Zero out BOS column
        if exclude_current_token:
            attn.fill_diagonal_(0)  # Zero out diagonal (current token attention)
        
        # Element-wise multiplication: only keep attention at expected positions
        score = attn * detection_pattern
        # Normalize by total attention: what fraction matches the pattern?
        return (score.sum() / attn.sum()).item() if attn.sum() > 0 else 0.0

    # abs: absolute difference (recommended)
    # Ensure both patterns are lower triangular (custom models may return full matrices)
    # Note: torch.tril() doesn't support BFloat16, so convert to float32 first
    attention_dtype = attention_pattern.dtype
    detection_dtype = detection_pattern.dtype
    
    # Convert to float32 for tril operation, then convert back
    attention_pattern = torch.tril(attention_pattern.float()).to(attention_dtype)
    detection_pattern = torch.tril(detection_pattern.float()).to(detection_dtype)
    
    # Compute element-wise absolute difference
    abs_diff = (attention_pattern - detection_pattern).abs()
    # Verify it's still lower triangular (should be, since both inputs are now)
    # Use a small tolerance instead of strict equality to handle floating point errors
    abs_diff_float = abs_diff.float() if abs_diff.dtype == torch.bfloat16 else abs_diff
    tril_diff = abs_diff_float - torch.tril(abs_diff_float).to(abs_diff_float.device)
    if abs_diff.dtype == torch.bfloat16:
        tril_diff = tril_diff.to(torch.bfloat16)
    if tril_diff.abs().sum() > 1e-6:
        # If there's significant difference, warn but continue
        logging.warning(
            f"Attention pattern may not be strictly lower triangular. "
            f"Upper triangular sum: {tril_diff.abs().sum().item():.6f}"
        )

    size = len(abs_diff)
    if exclude_bos:
        abs_diff[:, 0] = 0  # Ignore BOS column in error calculation
    if exclude_current_token:
        abs_diff.fill_diagonal_(0)  # Ignore diagonal in error calculation

    # Convert error to similarity score
    # mean_abs_diff ranges from 0 to 2 (max difference)
    # similarity = 1 - mean_error gives range [-1, 1]
    # where 1 = perfect match, -1 = complete mismatch
    mean_error = abs_diff.mean().item()
    similarity = 1 - mean_error
    return similarity


# ============================================================================
# Main detection function
# ============================================================================

def detect_head(
    model: CustomModelAdapter,
    seq: Union[str, List[str]],
    detection_pattern: Union[torch.Tensor, HeadName],
    heads: Optional[Union[List[LayerHeadTuple], LayerToHead]] = None,
    cache: Optional[ActivationCache] = None,
    *,
    exclude_bos: bool = False,
    exclude_current_token: bool = False,
    error_measure: ErrorMeasure = "mul",
) -> torch.Tensor:
    """Searches the model for a particular type of attention head.
    
    This function searches for attention heads that match a specific attention pattern.
    The detection pattern can be:
    1. A pre-defined head name (e.g., "previous_token_head")
    2. A custom tensor representing the expected attention pattern
    
    There are two error measures:
    - "mul" (default): Element-wise multiplication. Good for binary patterns (0/1).
    - "abs" (recommended): Absolute difference. Better for precise predictions.
    
    Args:
        model: CustomModelAdapter model being used.
        seq: String or list of strings being fed to the model.
        detection_pattern: Either a head name string or a (seq_len, seq_len) tensor.
                          Available head names: ["previous_token_head", "duplicate_token_head", 
                          "induction_head"]
        heads: Optional. If provided, only check these specific heads. Can be:
              - List of (layer, head) tuples
              - Dict mapping layer -> list of heads
        cache: Optional. Pre-computed activation cache to save time.
        exclude_bos: Exclude attention paid to the beginning-of-sequence token.
        exclude_current_token: Exclude attention paid to the current token.
        error_measure: "mul" for multiplication, "abs" for absolute difference.

    Returns:
        A (n_layers, n_heads) Tensor representing the score for each attention head.
        For induction_head: scores range from 0 to 1 (mean attention weight on induction positions).
        For other heads: scores range from -1 (mismatch) to 1 (perfect match).

    Example:
        >>> from model_adapter import CustomModelAdapter
        >>> from head_detect_llama_custom import detect_head
        >>> import torch
        >>> 
        >>> # Load custom model
        >>> model = CustomModelAdapter.from_pretrained(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     device="cuda",
        ...     torch_dtype=torch.bfloat16,
        ...     use_flash_attention_2=False
        ... )
        >>> 
        >>> sequence = "This is a test sequence. This is a test sequence."
        >>> 
        >>> # Detect previous token heads
        >>> scores = detect_head(model, sequence, "previous_token_head")
        >>> print(scores.shape)  # (n_layers, n_heads)
        >>> 
        >>> # Detect with custom options
        >>> scores = detect_head(
        ...     model, 
        ...     sequence, 
        ...     "previous_token_head",
        ...     exclude_bos=True, 
        ...     exclude_current_token=True,
        ...     error_measure="abs"
        ... )
    """
    if not HAS_CUSTOM_MODEL:
        raise ImportError(
            "CustomModelAdapter is not available. "
            "Please ensure model_adapter.py and hook_utils.py are in the path."
        )
    
    cfg = model.cfg
    tokens = model.to_tokens(seq).to(cfg.device)
    seq_len = tokens.shape[-1]
    
    # Validate error_measure
    assert error_measure in get_args(ErrorMeasure), (
        f"Invalid {error_measure=}; valid values are {get_args(ErrorMeasure)}"
    )

    # Validate detection pattern if it's a string
    if isinstance(detection_pattern, str):
        assert detection_pattern in HEAD_NAMES, (
            INVALID_HEAD_NAME_ERR % detection_pattern
        )
        if isinstance(seq, list):
            # Handle batch of sequences: compute scores for each, then average
            batch_scores = [
                detect_head(
                    model, s, detection_pattern, heads=heads, 
                    cache=cache, exclude_bos=exclude_bos, 
                    exclude_current_token=exclude_current_token,
                    error_measure=error_measure
                ) for s in seq
            ]
            return torch.stack(batch_scores).mean(0)
        
        # Special handling for induction_head: use original TransformerLens method
        # (directly extract diagonal instead of pattern matching)
        if detection_pattern == "induction_head":
            # Get cache if not provided
            if cache is None:
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            # Process heads
            if heads is None:
                layer2heads = {
                    layer_i: list(range(cfg.n_heads)) for layer_i in range(cfg.n_layers)
                }
            elif isinstance(heads, list):
                layer2heads = defaultdict(list)
                for layer, head in heads:
                    layer2heads[layer].append(head)
            else:
                layer2heads = heads
            
            # Initialize scores with zeros (range will be [0, 1])
            matches = torch.zeros(cfg.n_layers, cfg.n_heads, device=cfg.device)
            
            # For each layer and head, extract diagonal and compute mean
            for layer, layer_heads in layer2heads.items():
                # Get attention patterns for all heads in this layer
                # Shape: [n_heads, q_pos, k_pos]
                layer_attention_patterns = cache["pattern", layer, "attn"]
                
                for head in layer_heads:
                    # Extract attention pattern for this specific head
                    # Shape: [q_pos, k_pos]
                    head_attention_pattern = layer_attention_patterns[head, :, :]
                    
                    # Extract diagonal: attention paid from each position to seq_len-1 tokens back
                    # This corresponds to induction head behavior: attending to token after duplicate
                    induction_stripe = head_attention_pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
                    
                    # Compute mean score (range: [0, 1])
                    # Higher score means more attention paid to induction positions
                    induction_score = induction_stripe.mean().item()
                    matches[layer, head] = induction_score
            
            return matches
        
        # Generate detection pattern from head name for other head types
        # For "previous_token_head", this calls get_previous_token_head_detection_pattern()
        # which creates a matrix where pattern[i, i-1] = 1 for all i > 0
        detection_pattern = cast(
            torch.Tensor,
            eval(f"get_{detection_pattern}_detection_pattern(tokens.cpu())"),
        ).to(cfg.device)

    # Warn if using "mul" with non-binary pattern
    if error_measure == "mul" and not set(detection_pattern.unique().tolist()).issubset(
        {0, 1}
    ):
        logging.warning(
            "Using detection pattern with values other than 0 or 1 with error_measure 'mul'"
        )

    # Validate inputs and detection pattern shape
    assert 1 < tokens.shape[-1] < cfg.n_ctx, SEQ_LEN_ERR
    assert (
        is_lower_triangular(detection_pattern) and seq_len == detection_pattern.shape[0]
    ), DET_PAT_NOT_SQUARE_ERR % (seq_len, detection_pattern.shape)

    # Get cache if not provided
    if cache is None:
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    # Process heads
    if heads is None:
        layer2heads = {
            layer_i: list(range(cfg.n_heads)) for layer_i in range(cfg.n_layers)
        }
    elif isinstance(heads, list):
        layer2heads = defaultdict(list)
        for layer, head in heads:
            layer2heads[layer].append(head)
    else:
        layer2heads = heads

    # Compute scores for each head
    # Initialize with -1 (no match) for all heads
    # Note: This is only reached for non-induction_head patterns
    matches = -torch.ones(cfg.n_layers, cfg.n_heads, device=cfg.device)

    # For each layer and head, compare actual attention pattern with detection pattern
    for layer, layer_heads in layer2heads.items():
        # Get attention patterns for all heads in this layer
        # Shape: [n_heads, q_pos, k_pos]
        layer_attention_patterns = cache["pattern", layer, "attn"]
        
        for head in layer_heads:
            # Extract attention pattern for this specific head
            # Shape: [q_pos, k_pos] - lower triangular matrix
            head_attention_pattern = layer_attention_patterns[head, :, :]
            
            # Compute similarity score between actual pattern and expected pattern
            # For previous_token_head:
            #   - detection_pattern has 1's at positions (i, i-1) for all i > 0
            #   - head_attention_pattern is the actual attention weights
            #   - score measures how well they match
            head_score = compute_head_attention_similarity_score(
                head_attention_pattern,
                detection_pattern=detection_pattern,
                exclude_bos=exclude_bos,
                exclude_current_token=exclude_current_token,
                error_measure=error_measure,
            )
            matches[layer, head] = head_score
    
    # Return matrix of shape (n_layers, n_heads) with similarity scores
    return matches


# ============================================================================
# Plotting utilities
# ============================================================================

def plot_head_detection_scores(
    scores: torch.Tensor,
    zmin: float = -1,
    zmax: float = 1,
    xaxis: str = "Head",
    yaxis: str = "Layer",
    title: str = "Head Matches"
) -> None:
    """Plot head detection scores as a heatmap.
    
    Args:
        scores: (n_layers, n_heads) Tensor of detection scores.
        zmin: Minimum value for color scale.
        zmax: Maximum value for color scale.
        xaxis: Label for x-axis.
        yaxis: Label for y-axis.
        title: Plot title.
    
    Raises:
        ImportError: If neel_plotly is not installed.
    """
    imshow(scores, zmin=zmin, zmax=zmax, xaxis=xaxis, yaxis=yaxis, title=title)


def plot_attn_pattern_from_cache(
    model: CustomModelAdapter,
    cache: ActivationCache, 
    prompt: str,
    layer_i: int
):
    """Plot attention patterns from cache for a specific layer.
    
    Args:
        model: The model being used.
        cache: Activation cache from model.run_with_cache().
        prompt: The prompt string used to generate the cache.
        layer_i: Layer index to plot.
    
    Returns:
        pysvelte.AttentionMulti object if pysvelte is available.
    
    Raises:
        ImportError: If pysvelte is not installed.
    """
    if not HAS_PYSVELTE:
        raise ImportError(
            "pysvelte is not installed. Install it with:\n"
            "  pip install git+https://github.com/neelnanda-io/PySvelte.git\n"
            "Or use --user flag:\n"
            "  pip install --user git+https://github.com/neelnanda-io/PySvelte.git\n"
            "Note: pysvelte requires Node.js. If installation fails, you may need to install Node.js first."
        )
    attention_pattern = cache["pattern", layer_i, "attn"].squeeze(0)
    attention_pattern = einops.rearrange(attention_pattern, "heads seq1 seq2 -> seq1 seq2 heads")
    print(f"Layer {layer_i} Attention Heads:")
    return pysvelte.AttentionMulti(tokens=model.to_str_tokens(prompt), attention=attention_pattern)


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device = }")
    
    if not HAS_CUSTOM_MODEL:
        print("❌ CustomModelAdapter is not available!")
        print("   Please ensure model_adapter.py and hook_utils.py are in the path.")
        exit(1)
    
    # Load model
    print("Loading custom model...")
    print("⚠️  Note: Disabling Flash Attention 2 for accurate attention weights")
    
    try:
        model = CustomModelAdapter.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            device=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            use_flash_attention_2=False,  # Disable Flash Attention 2 for accurate attention weights
        )
        print("✅ Successfully loaded custom LLaMA2-7B model")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Show supported heads
    print(f"\nSupported heads: {get_supported_heads()}")
    
    # Create output directory for saving head scores
    output_dir = Path("/home/jhe/Head_analysis/duplicate_head/demos/head_scores")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving head scores to: {output_dir}")
    
    # Example 1: Detect previous token head
    print("\n" + "="*60)
    print("Example 1: Detecting previous token heads")
    print("="*60)
    prompt = "The head detector feature for TransformerLens allows users to check for various common heads automatically, reducing the cost of discovery."
    head_scores = detect_head(model, prompt, "previous_token_head")
    print(f"Detection scores shape: {head_scores.shape}")
    print(f"Max score: {head_scores.max():.3f} at layer {head_scores.max(1)[0].argmax().item()}, head {head_scores.max(1)[1].argmax().item()}")
    
    # Save Example 1 scores
    save_path = output_dir / "previous_token_head_custom_default.pt"
    torch.save(head_scores, save_path)
    print(f"Saved: {save_path}")
    
    if HAS_NEEL_PLOTLY:
        plot_head_detection_scores(head_scores, title="Previous Head Matches (Custom Model)")
    else:
        print("(Skipping visualization - neel_plotly not installed)")
    
    # Example 2: Detect with custom options
    print("\n" + "="*60)
    print("Example 2: Detecting with exclude_bos and exclude_current_token")
    print("="*60)
    head_scores = detect_head(
        model, 
        prompt, 
        "previous_token_head", 
        exclude_bos=True, 
        exclude_current_token=True,
        error_measure="abs"
    )
    print(f"Max score: {head_scores.max():.3f}")
    
    # Save Example 2 scores
    save_path = output_dir / "previous_token_head_custom_excluded_bos_current_abs.pt"
    torch.save(head_scores, save_path)
    print(f"Saved: {save_path}")
    
    if HAS_NEEL_PLOTLY:
        plot_head_detection_scores(head_scores, title="Previous Head Matches (excluded BOS and current token, Custom Model)")
    else:
        print("(Skipping visualization - neel_plotly not installed)")
    
    # Example 3: Detect duplicate token head
    print("\n" + "="*60)
    print("Example 3: Detecting duplicate token heads")
    print("="*60)
    prompts = [
        "one two three one two three one two three",
        "1 2 3 4 5 1 2 3 4 1 2 3 1 2 3 4 5 6 7",
        "green ideas sleep furiously; green ideas don't sleep furiously"
    ]
    head_scores = detect_head(
        model, 
        prompts, 
        "duplicate_token_head", 
        exclude_bos=False, 
        exclude_current_token=False, 
        error_measure="abs"
    )
    print(f"Detection scores shape: {head_scores.shape}")
    print(f"Max score: {head_scores.max():.3f}")
    
    # Save Example 3 scores
    save_path = output_dir / "duplicate_token_head_custom_abs.pt"
    torch.save(head_scores, save_path)
    print(f"Saved: {save_path}")
    
    if HAS_NEEL_PLOTLY:
        plot_head_detection_scores(head_scores, title="Duplicate token head (Custom Model); average across 3 prompts")
    else:
        print("(Skipping visualization - neel_plotly not installed)")
    
    # Example 4: Detect induction head
    print("\n" + "="*60)
    print("Example 4: Detecting induction heads")
    print("="*60)
    # Use the same prompts as Example 3 for consistency
    head_scores = detect_head(
        model, 
        prompts, 
        "induction_head", 
        exclude_bos=False, 
        exclude_current_token=False, 
        error_measure="abs"
    )
    print(f"Detection scores shape: {head_scores.shape}")

    print(f"Max score: {head_scores.max():.3f} at layer {head_scores.max(1)[0].argmax().item()}, head {head_scores.max(1)[1].argmax().item()}")
    
    # Save Example 4 scores
    save_path = output_dir / "induction_head_custom_abs.pt"
    torch.save(head_scores, save_path)
    print(f"Saved: {save_path}")
    
    if HAS_NEEL_PLOTLY:
        plot_head_detection_scores(head_scores, title="Induction head (Custom Model); average across 3 prompts")
    else:
        print("(Skipping visualization - neel_plotly not installed)")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print(f"All head scores saved to: {output_dir}")
    print("="*60)

        
    # ============================================================================
    # Iteration Head Detection (using iteration_head_detector.py)
    # ============================================================================
    # Add Iteration_head/src to path for cot module
    _iteration_head_root = _current_dir.parent.parent / "Iteration_head"
    _iteration_head_src_dir = _iteration_head_root / "src"
    if str(_iteration_head_src_dir) not in sys.path:
        sys.path.insert(0, str(_iteration_head_src_dir))
    
    try:
        from iteration_head_detector import detect_iteration_heads
        from cot.config import TOKEN_DICT
        from cot.data import Parity
        HAS_ITERATION_HEAD = True
    except ImportError as e:
        HAS_ITERATION_HEAD = False
        print(f"\n⚠️  Warning: Could not import iteration head detection modules: {e}")
        print("   Skipping Iteration Head Detection section.")
        print(f"   Iteration_head/src path: {_iteration_head_src_dir}")
    
    if not HAS_ITERATION_HEAD:
        print("\n" + "="*60)
        print("Skipping Iteration Head Detection (modules not available)")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Iteration Head Detection: Data Preparation")
        print("="*60)

        # Configuration for iteration head detection
        problem = "parity"          # Can be "parity", "binary-copy", or "polynomial"
        n_len = 16                  # Maximum sequence length for the CoT problem
        data_type = "test"          # "train" or "test"
        data_dir = Path("/home/jhe/Head_analysis/duplicate_head/demos/data/iteration")
        n_samples = 100             # Number of samples to use (None = use all)
        sample_shuffle = True       # Shuffle before subsampling
        seed = 0                    # Seed for subsampling

        # Select problem class (here we only use Parity; extend if needed)
        ProblemClass = Parity

        # Create dataset and ensure data exists
        dataset = ProblemClass(save_dir=data_dir, cot=True)
        lengths = list(range(1, n_len + 1))

        # Generate data files if missing
        need_generate = False
        for seq_len in lengths:
            data_file = data_dir / f"{data_type}_{seq_len}.npy"
            if not data_file.exists():
                need_generate = True
                break

        if need_generate:
            print(f"Data files not found in {data_dir}, generating data...")
            n_data_per_len = [1000] * n_len
            split_probas_by_len = [0.8] * n_len  # 80% train, 20% test
            dataset.generate_datafiles(n_data_per_len, split_probas_by_len)
            print(f"Data generation completed. Data saved to {data_dir}")

        # Load data for specified lengths and type
        dataset.set_data(lengths, data_type=data_type)

        if n_samples is not None and len(dataset) > n_samples:
            if sample_shuffle:
                g = torch.Generator()
                g.manual_seed(seed)
                indices = torch.randperm(len(dataset), generator=g)[:n_samples]
            else:
                indices = torch.arange(n_samples)
            dataset.data = dataset.data[indices]

        sequences = dataset.data  # (batch_size, seq_len)
        print(f"Loaded {len(sequences)} sequences for iteration head detection.")

        # Detect iteration heads
        print("\n" + "="*60)
        print("Running Iteration Head Detection")
        print("="*60)

        head_triplets = detect_iteration_heads(
            model=model,
            model_name="meta-llama/Llama-2-7b-hf",
            sequences=sequences,
            token_dict=TOKEN_DICT,
            peaky_threshold=0.5,
            inv_threshold=0.7,
            batch_size=1,
        )

        # Save results to disk for later visualization
        if head_triplets.size > 0:
            iter_save_path = output_dir / "iteration_heads_inv_gt_0.70_sorted.npy"
            np.save(iter_save_path, head_triplets)
            print(f"\nSaved iteration heads (layer, head, peaky) to: {iter_save_path}")
        else:
            print("\nNo iteration-like heads found with inv > 0.70.")


