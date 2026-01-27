"""
Hook utilities for custom models to capture activations similar to TransformerLens.

This module provides functions to add hooks to custom LlamaForCausalLM models,
allowing you to capture all activations while still using attn_mode="torch" to get attention weights.
"""

import torch
from typing import Dict, List, Optional, Callable
from collections import defaultdict


class ActivationCache:
    """Simple activation cache similar to TransformerLens."""
    
    def __init__(self, cache_dict: Dict[str, torch.Tensor]):
        self.cache_dict = cache_dict
    
    def __getitem__(self, key):
        """Support both dict access and tuple access like TransformerLens."""
        if isinstance(key, tuple):
            # Support TransformerLens-style access: cache["pattern", layer, "attn"]
            if len(key) == 3 and key[0] == "pattern" and key[2] == "attn":
                layer = key[1]
                # Get attention pattern for this layer
                attn_key = f"blocks.{layer}.attn.hook_pattern"
                if attn_key in self.cache_dict:
                    attn = self.cache_dict[attn_key]
                    
                    # Handle None case (processing failed)
                    if attn is None:
                        return None
                    
                    # Handle different possible shapes:
                    # 1. [batch, n_heads, seq_len, seq_len] - with batch dimension
                    # 2. [n_heads, seq_len, seq_len] - already removed batch dimension
                    
                    # For backward compatibility with TransformerLens, remove batch dimension if batch_size=1
                    # But preserve 4D format if batch_size > 1 (for iteration_head_detector)
                    if attn.dim() == 4:
                        # Shape: [batch, n_heads, seq_len, seq_len]
                        if attn.shape[0] == 1:
                            # Remove batch dimension for TransformerLens compatibility
                            attn = attn.squeeze(0)  # Remove batch dimension: [n_heads, seq_len, seq_len]
                        else:
                            # Multiple batches - return as is (4D format)
                            # This is valid when remove_batch_dim=False
                            return attn
                    elif attn.dim() == 3:
                        # Already in format [n_heads, seq_len, seq_len]
                        # This is the expected format after remove_batch_dim
                        pass
                    else:
                        # Unexpected shape
                        raise ValueError(f"Unexpected attention pattern shape: {attn.shape}")
                    
                    # Final format should be [n_heads, seq_len, seq_len]
                    # This matches TransformerLens format
                    return attn
            # Support other tuple formats
            key_str = ".".join(str(k) for k in key)
            return self.cache_dict.get(key_str)
        return self.cache_dict.get(key)
    
    def keys(self):
        return self.cache_dict.keys()


def add_hooks_to_model(
    model: torch.nn.Module,
    cache: Optional[Dict[str, torch.Tensor]] = None,
    remove_batch_dim: bool = True
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Add forward hooks to a custom LlamaForCausalLM model to capture activations.
    
    Args:
        model: The custom LlamaForCausalLM model
        cache: Dictionary to store activations (will create new one if None)
        remove_batch_dim: If True, remove batch dimension when batch_size=1
    
    Returns:
        List of hook handles that can be used to remove hooks later
    """
    if cache is None:
        cache = {}
    
    hooks = []
    
    # Hook function to save activations
    def make_hook(name: str):
        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, torch.Tensor):
                tensor = output.detach()
                if remove_batch_dim and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                cache[name] = tensor
            elif isinstance(output, tuple):
                # For tuple outputs, save the first element (usually the main tensor)
                if len(output) > 0 and isinstance(output[0], torch.Tensor):
                    tensor = output[0].detach()
                    if remove_batch_dim and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    cache[name] = tensor
        return hook_fn
    
    # Hook embedding layer
    # Try different possible paths for embedding (LLaMA and GPT-NeoX)
    embed_hooked = False
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # LLaMA style: model.model.embed_tokens
        hooks.append(
            model.model.embed_tokens.register_forward_hook(
                make_hook("embed")
            )
        )
        embed_hooked = True
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_in'):
        # GPT-NeoX style: model.model.embed_in
        hooks.append(
            model.model.embed_in.register_forward_hook(
                make_hook("embed")
            )
        )
        embed_hooked = True
    elif hasattr(model, 'embed_tokens'):
        # Some models might have embed_tokens directly
        hooks.append(
            model.embed_tokens.register_forward_hook(
                make_hook("embed")
            )
        )
        embed_hooked = True
    
    if not embed_hooked:
        # Log warning but continue - embed might not be critical for all use cases
        import warnings
        warnings.warn(
            "Could not find embedding layer to hook. "
            "Model structure might be different. "
            "Available attributes: " + str([attr for attr in dir(model) if 'embed' in attr.lower()])
        )
    
    # Hook each decoder layer
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer_idx, layer in enumerate(model.model.layers):
            # Hook input layernorm (LLaMA: input_layernorm, GPT-NeoX: input_layernorm)
            if hasattr(layer, 'input_layernorm'):
                hooks.append(
                    layer.input_layernorm.register_forward_hook(
                        make_hook(f"blocks.{layer_idx}.ln1.hook_normalized")
                    )
                )
            
            # Hook attention module
            if hasattr(layer, 'self_attn'):
                # Hook Q, K, V projections to compute attention patterns manually if needed
                # This provides a backup if outputs.attentions is incorrect
                # LLaMA: q_proj, k_proj, v_proj
                # GPT-NeoX: query_key_value (combined projection)
                if hasattr(layer.self_attn, 'q_proj'):
                    # LLaMA style: separate projections
                    hooks.append(
                        layer.self_attn.q_proj.register_forward_hook(
                            make_hook(f"blocks.{layer_idx}.attn.hook_q")
                        )
                    )
                if hasattr(layer.self_attn, 'k_proj'):
                    hooks.append(
                        layer.self_attn.k_proj.register_forward_hook(
                            make_hook(f"blocks.{layer_idx}.attn.hook_k")
                        )
                    )
                if hasattr(layer.self_attn, 'v_proj'):
                    hooks.append(
                        layer.self_attn.v_proj.register_forward_hook(
                            make_hook(f"blocks.{layer_idx}.attn.hook_v")
                        )
                    )
                elif hasattr(layer.self_attn, 'query_key_value'):
                    # GPT-NeoX style: combined projection
                    hooks.append(
                        layer.self_attn.query_key_value.register_forward_hook(
                            make_hook(f"blocks.{layer_idx}.attn.hook_qkv")
                        )
                    )
                
                # Hook attention output
                hooks.append(
                    layer.self_attn.register_forward_hook(
                        make_hook(f"blocks.{layer_idx}.attn.hook_result")
                    )
                )
            
            # Hook post-attention layernorm
            # LLaMA: post_attention_layernorm
            # GPT-NeoX: post_attention_layernorm (same name)
            if hasattr(layer, 'post_attention_layernorm'):
                hooks.append(
                    layer.post_attention_layernorm.register_forward_hook(
                        make_hook(f"blocks.{layer_idx}.ln2.hook_normalized")
                    )
                )
            
            # Hook MLP
            if hasattr(layer, 'mlp'):
                hooks.append(
                    layer.mlp.register_forward_hook(
                        make_hook(f"blocks.{layer_idx}.mlp.hook_out")
                    )
                )
            
            # Hook residual stream (layer output)
            hooks.append(
                layer.register_forward_hook(
                    make_hook(f"blocks.{layer_idx}.hook_resid_post")
                )
            )
    
    # Hook final layer norm
    # LLaMA: model.model.norm
    # GPT-NeoX: model.model.final_layer_norm
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        hooks.append(
            model.model.norm.register_forward_hook(
                make_hook("ln_final.hook_normalized")
            )
        )
    elif hasattr(model, 'model') and hasattr(model.model, 'final_layer_norm'):
        hooks.append(
            model.model.final_layer_norm.register_forward_hook(
                make_hook("ln_final.hook_normalized")
            )
        )
    
    return hooks


def hook_attention_patterns(
    model: torch.nn.Module,
    cache: Dict[str, torch.Tensor],
    remove_batch_dim: bool = True
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Specifically hook attention patterns from forward_torch method.
    This requires modifying the attention forward to capture patterns.
    
    Note: This is more complex because we need to intercept the attention computation.
    For now, we rely on output_attentions=True and attn_mode="torch" to get patterns.
    """
    hooks = []
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, 'self_attn'):
                # We'll capture attention patterns from the model's output
                # when output_attentions=True
                pass
    
    return hooks


def run_with_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: Optional[Dict[str, torch.Tensor]] = None,
    remove_batch_dim: bool = True,
    output_attentions: bool = True,
    attn_mode: str = "torch",
    **kwargs
) -> tuple:
    """
    Run model with caching, similar to TransformerLens's run_with_cache.
    
    Args:
        model: Custom LlamaForCausalLM model
        input_ids: Input token ids
        cache: Optional cache dict (will create new one if None)
        remove_batch_dim: Remove batch dimension if batch_size=1
        output_attentions: Whether to output attention weights
        attn_mode: "torch" to get attention weights, "flash" for speed
        **kwargs: Additional arguments to pass to model
    
    Returns:
        Tuple of (logits, cache_dict, attentions)
    """
    if cache is None:
        cache = {}
    
    # Add hooks
    hooks = add_hooks_to_model(model, cache, remove_batch_dim)
    
    try:
        # Run model with output_attentions to get attention patterns
        with torch.no_grad():
            # Check if model supports attn_mode parameter
            # Custom models (faiss_attn) support it, but standard HuggingFace models don't
            model_kwargs = {
                'input_ids': input_ids,
                'output_attentions': output_attentions,
                'return_dict': True,
                **kwargs
            }
            
            # Try to determine if model supports attn_mode
            # Check if the model has custom attention implementation
            supports_attn_mode = False
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                # Check first layer's attention module
                first_layer = model.model.layers[0]
                if hasattr(first_layer, 'self_attn'):
                    # Custom models have forward_torch method
                    if hasattr(first_layer.self_attn, 'forward_torch'):
                        supports_attn_mode = True
                    # Or check if it's a custom attention class
                    elif hasattr(first_layer.self_attn, '__class__'):
                        class_name = first_layer.self_attn.__class__.__name__
                        # Custom attention classes might have different names
                        # But if it's standard LlamaAttention, it won't have forward_torch
                        if 'LlamaAttention' in class_name:
                            # Check if it has forward_torch (custom) or not (standard)
                            supports_attn_mode = hasattr(first_layer.self_attn, 'forward_torch')
            
            # Only add attn_mode if model supports it
            if supports_attn_mode and attn_mode is not None:
                model_kwargs['attn_mode'] = attn_mode
            
            # Try to call model, with fallback if attn_mode causes issues
            try:
                outputs = model(**model_kwargs)
            except (AttributeError, TypeError) as e:
                # If error is related to forward_torch or attn_mode, try without it
                if 'forward_torch' in str(e) or 'attn_mode' in str(e):
                    if 'attn_mode' in model_kwargs:
                        import warnings
                        warnings.warn(
                            f"Model does not support attn_mode parameter. "
                            f"Removing it and using standard forward pass. "
                            f"Error: {e}"
                        )
                        del model_kwargs['attn_mode']
                        outputs = model(**model_kwargs)
                    else:
                        raise
                else:
                    raise
        
        logits = outputs.logits
        
        # Store attention patterns in cache
        attentions = None
        if output_attentions and hasattr(outputs, 'attentions'):
            attentions = outputs.attentions
            
            # Validate and fix attention patterns
            if attentions is not None and len(attentions) > 0:
                # Store each layer's attention pattern
                for layer_idx, attn in enumerate(attentions):
                    # Validate attention tensor
                    if attn is None:
                        import warnings
                        warnings.warn(
                            f"Layer {layer_idx}: outputs.attentions[{layer_idx}] is None. "
                            f"This may indicate that attention weights are not available."
                        )
                        continue
                    
                    # attn shape should be [batch, n_heads, seq_len, seq_len]
                    attn_tensor = attn.detach()
                    
                    # Validate shape
                    if attn_tensor.dim() != 4:
                        import warnings
                        warnings.warn(
                            f"Layer {layer_idx}: Unexpected attention shape {attn_tensor.shape}. "
                            f"Expected [batch, n_heads, seq_len, seq_len]"
                        )
                        # Try to fix: if it's 3D, add batch dimension
                        if attn_tensor.dim() == 3:
                            attn_tensor = attn_tensor.unsqueeze(0)
                    
                    # Validate values: check if attention weights are normalized
                    # Each row should sum to approximately 1.0
                    if attn_tensor.dim() == 4:
                        batch_size, n_heads, seq_len, _ = attn_tensor.shape
                        # Check normalization for first batch and first head
                        sample_attn = attn_tensor[0, 0, :, :].float()
                        row_sums = sample_attn.sum(dim=-1)
                        mean_sum = row_sums.mean().item()
                        
                        # If attention weights are not normalized (sum != 1.0), they might be raw scores
                        # In this case, we need to apply softmax
                        if abs(mean_sum - 1.0) > 0.1:  # Not normalized
                            import warnings
                            warnings.warn(
                                f"Layer {layer_idx}: Attention weights are not normalized "
                                f"(mean row sum={mean_sum:.6f}, expected ~1.0). "
                                f"Applying softmax to normalize."
                            )
                            # Apply softmax to normalize
                            # Shape: [batch, n_heads, seq_len, seq_len]
                            attn_tensor = torch.nn.functional.softmax(attn_tensor.float(), dim=-1).to(attn_tensor.dtype)
                        
                        # Check for NaN or Inf
                        if torch.isnan(attn_tensor).any() or torch.isinf(attn_tensor).any():
                            import warnings
                            warnings.warn(
                                f"Layer {layer_idx}: Attention weights contain NaN or Inf. "
                                f"This may indicate a problem with the model."
                            )
                            # Replace NaN/Inf with zeros
                            attn_tensor = torch.where(
                                torch.isnan(attn_tensor) | torch.isinf(attn_tensor),
                                torch.zeros_like(attn_tensor),
                                attn_tensor
                            )
                    
                    # Remove batch dimension if needed
                    # Only remove if it's 4D and batch_size == 1
                    if remove_batch_dim and attn_tensor.dim() == 4 and attn_tensor.shape[0] == 1:
                        attn_tensor = attn_tensor.squeeze(0)
                    
                    # Final validation: accept both 3D and 4D shapes
                    # 3D: [n_heads, seq_len, seq_len] (when remove_batch_dim=True)
                    # 4D: [batch, n_heads, seq_len, seq_len] (when remove_batch_dim=False)
                    if attn_tensor.dim() == 3:
                        # Store in TransformerLens-compatible format (3D)
                        cache[f"blocks.{layer_idx}.attn.hook_pattern"] = attn_tensor
                    elif attn_tensor.dim() == 4:
                        # Store with batch dimension (4D) - this is valid when remove_batch_dim=False
                        cache[f"blocks.{layer_idx}.attn.hook_pattern"] = attn_tensor
                    else:
                        import warnings
                        warnings.warn(
                            f"Layer {layer_idx}: Failed to process attention pattern. "
                            f"Final shape: {attn_tensor.shape}, expected [n_heads, seq_len, seq_len] "
                            f"or [batch, n_heads, seq_len, seq_len]"
                        )
                        
                        # Try to compute attention pattern from Q, K if available
                        q_key = f"blocks.{layer_idx}.attn.hook_q"
                        k_key = f"blocks.{layer_idx}.attn.hook_k"
                        if q_key in cache and k_key in cache:
                            try:
                                # Compute attention pattern from Q and K
                                Q = cache[q_key].float()
                                K = cache[k_key].float()
                                
                                # Get model config for scaling
                                if hasattr(model, 'config'):
                                    head_dim = getattr(model.config, 'head_dim', 
                                                      getattr(model.config, 'd_model', 4096) // 
                                                      getattr(model.config, 'num_attention_heads', 32))
                                else:
                                    head_dim = 128  # Default for LLaMA-2-7B
                                
                                # Reshape Q and K: [seq_len, d_model] -> [n_heads, seq_len, d_head]
                                # This depends on the actual shape
                                if Q.dim() == 2 and K.dim() == 2:
                                    # Need to reshape based on model config
                                    # For now, skip this fallback as it requires model-specific knowledge
                                    pass
                                
                                import warnings
                                warnings.warn(
                                    f"Layer {layer_idx}: Attempted to compute attention from Q/K, "
                                    f"but shape conversion is model-specific. "
                                    f"Using outputs.attentions instead."
                                )
                            except Exception as e:
                                import warnings
                                warnings.warn(
                                    f"Layer {layer_idx}: Failed to compute attention from Q/K: {e}"
                                )
                        
                        # Store None to indicate failure, but don't crash
                        cache[f"blocks.{layer_idx}.attn.hook_pattern"] = None
            else:
                import warnings
                warnings.warn(
                    "outputs.attentions is None or empty. "
                    "Attention weights may not be available. "
                    "This could be due to Flash Attention 2 or model configuration. "
                    "Try setting use_flash_attention_2=False."
                )
        
        # Also store in the output for easy access
        cache['attentions'] = attentions if output_attentions else None
        
        return logits, ActivationCache(cache), attentions if output_attentions else None
        
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()


# Example usage:
if __name__ == "__main__":
    """
    Example of how to use hooks with custom model:
    
    from source.modeling_llama import LlamaForCausalLM
    from hook_utils import run_with_cache
    
    # Load your custom model
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        use_flash_attention_2="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map='auto'
    ).eval()
    
    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt")['input_ids']
    
    # Run with cache
    logits, cache, attentions = run_with_cache(
        model,
        input_ids,
        output_attentions=True,
        attn_mode="torch"  # Use torch mode to get attention weights
    )
    
    # Access activations like TransformerLens
    embed = cache["embed"]
    layer_0_attn = cache["pattern", 0, "attn"]  # TransformerLens-style access
    layer_0_resid = cache["blocks.0.hook_resid_post"]
    
    # Or access directly
    layer_0_attn_direct = cache.cache_dict["blocks.0.attn.hook_pattern"]
    """
    pass

