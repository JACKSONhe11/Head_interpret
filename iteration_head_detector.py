"""
Utilities for detecting iteration heads on top of a HookedTransformer model.

This factors out the \"iteration head\" scoring logic from
`head_detect_llama_jx_iteration.py` into a reusable function.

Main entry point:
    detect_iteration_heads(
        model,
        model_name,
        sequences,
        token_dict,
        peaky_threshold=0.5,
        inv_threshold=0.7,
        batch_size=1,
    ) -> np.ndarray

Returns a numpy array of shape (n_selected_heads, 3):
    [layer_index, head_index, iteration_peaky_score]
sorted by iteration_peaky in descending order, where each selected head has
average invariance > inv_threshold.
"""

from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def _wrap_prompt_for_model(prompt: str, model_name: str) -> str:
    """
    Wrap prompt according to model type (base vs chat/instruct).
    This is a lightweight copy of the logic used in the main demo script.
    """
    # LLaMA 2 chat style
    if "Llama-2" in model_name and "chat" in model_name:
        return (
            "[INST] <<SYS>>You are a helpful assistant that computes XOR parity. "
            "Only output the running parity sequence in the specified format.<</SYS>>\n\n"
            f"{prompt}\n[/INST]"
        )
    # LLaMA 3 Instruct style
    if "Llama-3" in model_name and ("Instruct" in model_name or "instruct" in model_name):
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant that computes XOR parity. "
            "Only output the running parity sequence in the specified format.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
    # Default: base models / GPT-2 etc.
    return prompt


def _token_ids_to_prompt(
    token_ids: List[int],
    token_dict,
    model,
    model_name: str,
) -> str:
    """
    Convert CoT token IDs to a natural language prompt for the LLM.

    This mirrors the logic in `head_detect_llama_jx_iteration.py` but is kept
    self-contained and uses only `token_dict` and `model`.
    """
    # Build mapping from special token IDs to strings (0/1 and task labels)
    bos_token = model.tokenizer.bos_token if model.tokenizer.bos_token else ""
    eos_token = model.tokenizer.eos_token if model.tokenizer.eos_token else ""

    TOKEN_ID_TO_STRING = {
        0: "0",
        1: "1",
        58: "polynomial",
        59: "parity",
        60: "binary_copy",
        int(token_dict.get("BoS", 61)): bos_token,
        int(token_dict.get("EoI", 62)): "EoI",
        int(token_dict.get("EoS", 63)): eos_token,
    }

    eoi_id = int(token_dict["EoI"])

    # Find EoI position
    eoi_pos = -1
    for i, tid in enumerate(token_ids):
        if tid == eoi_id:
            eoi_pos = i
            break

    if eoi_pos == -1:
        # Fallback: simple space-joined sequence
        tokens = []
        for tid in token_ids:
            if tid in TOKEN_ID_TO_STRING and TOKEN_ID_TO_STRING[tid]:
                tokens.append(TOKEN_ID_TO_STRING[tid])
            else:
                tokens.append(str(int(tid)))
        return " ".join(tokens)

    # Extract input (before EoI, skip BoS at position 0)
    input_tokens = token_ids[1:eoi_pos]
    input_bits = [str(int(tid)) if tid in [0, 1] else str(int(tid)) for tid in input_tokens]
    input_sequence = ", ".join(input_bits)

    base_prompt = f"""You are given a binary input sequence:

{input_sequence}

Define parity using XOR.
Assume the initial parity is 0 (even parity).

For each input bit, update the parity as:
parity_t = parity_{{t-1}} XOR input_t

Task:
Output the running parity sequence (one parity value per input bit).

Output format:
- Running parity: <comma-separated list>

Please provide only the running parity sequence. Do not include the final parity or any explanations, examples, or additional text."""

    # For now we keep example / hint flags off to keep the interface simple.

    return _wrap_prompt_for_model(base_prompt, model_name)


def detect_iteration_heads(
    model,
    model_name: str,
    sequences: torch.Tensor,
    token_dict,
    peaky_threshold: float = 0.5,
    inv_threshold: float = 0.7,
    batch_size: int = 1,
) -> np.ndarray:
    """
    Detect iteration heads for a given model and CoT dataset.

    Args
    ----
    model:
        HookedTransformer (or similar) model with `.to_tokens`, `.run_with_cache`,
        and `.tokenizer`.
    model_name: str
        Model name string (used only to decide chat/instruct wrapping style).
    sequences: torch.Tensor
        CoT token sequences of shape (batch_size, seq_len), using the same
        token IDs as `token_dict`.
    token_dict:
        Dictionary with at least entries for \"EoI\" and \"EoS\" used to locate
        the input/output split in sequences.
    peaky_threshold: float
        Threshold (on fraction of attention >0.5) for the EoI locator metric.
        Used only for reporting / optional filtering.
    inv_threshold: float
        Threshold on the invariance metric (1 - std) used to select iteration heads.
    batch_size: int
        Batch size for running the model when extracting attention patterns.

    Returns
    -------
    head_triplets: np.ndarray
        Array of shape (n_selected_heads, 3), sorted by iteration_peaky desc:
            [layer_index, head_index, iteration_peaky_score]
        Only heads with avg_iteration_inv > inv_threshold are included.
    """
    device = model.cfg.device if hasattr(model, "cfg") else "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("Iteration Head Detection: Extracting Attention Patterns")
    print("=" * 60)

    # Prepare prompts from CoT sequences
    all_attentions = []
    all_sequences = []
    all_seq_lengths: List[int] = []

    print(f"\nProcessing {len(sequences)} sequences...")
    for start in tqdm(range(0, len(sequences), batch_size), desc="Extracting"):
        batch_seqs = sequences[start:start + batch_size]
        batch_strings = [
            _token_ids_to_prompt(seq.tolist(), token_dict, model, model_name)
            for seq in batch_seqs
        ]

        with torch.no_grad():
            batch_tokens = model.to_tokens(batch_strings).to(device)
            _, cache = model.run_with_cache(batch_tokens, remove_batch_dim=False)

            # Collect attention patterns: (n_layers, batch, n_heads, seq_len, seq_len)
            batch_attentions = []
            for layer in range(model.cfg.n_layers):
                attn = cache["pattern", layer, "attn"]
                if attn is None:
                    # If attention pattern is None, skip this batch
                    import warnings
                    warnings.warn(
                        f"Layer {layer}: Attention pattern is None. "
                        f"This may indicate a problem with attention extraction."
                    )
                    # Use zeros as fallback
                    # Get shape from cache or use default
                    if batch_attentions:
                        # Use shape from previous layer
                        prev_shape = batch_attentions[-1].shape
                        attn = torch.zeros(prev_shape, device=batch_tokens.device, dtype=torch.float32)
                    else:
                        # First layer failed - use default shape
                        seq_len = batch_tokens.shape[1]
                        n_heads = model.cfg.n_heads
                        attn = torch.zeros((n_heads, seq_len, seq_len), device=batch_tokens.device, dtype=torch.float32)
                # Ensure 4D format: [batch, n_heads, seq_len, seq_len]
                if attn.dim() == 3:
                    # Add batch dimension
                    attn = attn.unsqueeze(0)
                batch_attentions.append(attn)
            batch_attentions = torch.stack(batch_attentions, dim=0)

        all_attentions.append(batch_attentions.cpu())
        all_seq_lengths.append(batch_tokens.shape[1])
        all_sequences.append(batch_seqs.cpu())

    # Pad to max sequence length if needed
    max_seq_len = max(all_seq_lengths) if all_seq_lengths else 0
    print(
        f"Tokenized lengths: min={min(all_seq_lengths)}, "
        f"max={max_seq_len}, mean={sum(all_seq_lengths) / len(all_seq_lengths):.1f}"
    )

    if len(set(all_seq_lengths)) > 1:
        print(f"Padding to max length: {max_seq_len}")
        padded_attentions = []
        for batch_attn in all_attentions:
            n_layers, batch, n_heads, seq_len, _ = batch_attn.shape
            if seq_len < max_seq_len:
                padded = torch.zeros(n_layers, batch, n_heads, max_seq_len, max_seq_len)
                padded[:, :, :, :seq_len, :seq_len] = batch_attn
                padded_attentions.append(padded)
            else:
                padded_attentions.append(batch_attn)
        all_attentions = padded_attentions

    # Concatenate along batch dimension
    attentions = torch.cat(all_attentions, dim=1)  # (n_layers, total_batch, n_heads, max_seq_len, max_seq_len)
    sequences_for_analysis = torch.cat(all_sequences, dim=0)  # (total_batch, seq_len)
    print(f"Sequences for analysis shape: {sequences_for_analysis.shape}")
    print(f"Attention shape: {attentions.shape}")

    # -------------------------------------------------------------------------
    # Compute iteration head scores (invariance + peakiness)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Computing Iteration Head Scores")
    print("=" * 60)

    # Find EoI positions in CoT sequences
    eoi_id = int(token_dict["EoI"])
    eois = torch.argmax((sequences_for_analysis == eoi_id).to(int), dim=1)
    all_eois = torch.unique(eois)
    print(f"Found {len(all_eois)} unique EoI positions: {all_eois.tolist()}")

    n_layers, n_batch, n_heads, seq_len, _ = attentions.shape
    n_lengths = len(all_eois)

    # [layer, head, length, metric]
    attn_inv = torch.zeros((n_layers, n_heads, n_lengths, 2))
    attn_peaky = torch.zeros((n_layers, n_heads, n_lengths, 4))

    for length_idx, eoi in enumerate(all_eois):
        eoi = eoi.item()
        ind = eois == eoi
        if ind.sum() == 0:
            continue

        eos = 2 * eoi  # EoS position in CoT indices

        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                layer_attn = attentions[layer_idx, ind, head_idx, :, :]  # (n_samples, seq_len, seq_len)

                # Pattern 1: CoT positions → EoI (EoI locator)
                if eoi + 1 < eos and eoi < seq_len:
                    attn_to_eoi = layer_attn[:, eoi + 1:eos, eoi]  # (n_samples, eos-eoi-1)
                    if attn_to_eoi.numel() > 0:
                        attn_inv[layer_idx, head_idx, length_idx, 0] = 1 - attn_to_eoi.std(dim=0).mean()
                        attn_peaky[layer_idx, head_idx, length_idx, 0] = attn_to_eoi.mean()
                        attn_peaky[layer_idx, head_idx, length_idx, 1] = (
                            (attn_to_eoi > 0.5).to(dtype=float).mean()
                        )

                # Pattern 2: diagonal iteration pattern
                if eoi > 1 and eoi < eos - 1 and eoi < seq_len:
                    attn_slice = layer_attn[:, eoi:eos - 1, 1:eoi]  # (n_samples, eos-1-eoi, eoi-1)
                    if attn_slice.numel() > 0:
                        n_diag = min(attn_slice.shape[1], attn_slice.shape[2])
                        if n_diag > 0:
                            attn_diagonal = torch.diagonal(attn_slice, dim1=1, dim2=2)  # (n_samples, n_diag)
                            if attn_diagonal.numel() > 0:
                                attn_inv[layer_idx, head_idx, length_idx, 1] = (
                                    1 - attn_diagonal.std(dim=0).mean()
                                )
                                attn_peaky[layer_idx, head_idx, length_idx, 2] = attn_diagonal.mean()
                                attn_peaky[layer_idx, head_idx, length_idx, 3] = (
                                    (attn_diagonal > 0.5).to(dtype=float).mean()
                                )

    scores = {
        "attn_eoi_inv": attn_inv[:, :, :, 0].numpy(),
        "attn_iteration_inv": attn_inv[:, :, :, 1].numpy(),
        "attn_eoi_peaky_abs": attn_peaky[:, :, :, 0].numpy(),
        "attn_eoi_peaky_thres": attn_peaky[:, :, :, 1].numpy(),
        "attn_iteration_peaky_abs": attn_peaky[:, :, :, 2].numpy(),
        "attn_iteration_peaky_thres": attn_peaky[:, :, :, 3].numpy(),
        "lengths": all_eois.numpy(),
    }

    print("Iteration head scores computed successfully")

    # -------------------------------------------------------------------------
    # Identify iteration heads based on invariance + (optionally) peaky
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Identifying Iteration Heads")
    print("=" * 60)

    iteration_scores = scores["attn_iteration_peaky_thres"]  # (n_layers, n_heads, n_lengths)
    iteration_inv = scores["attn_iteration_inv"]

    avg_iteration_peaky = np.mean(iteration_scores, axis=2)  # (n_layers, n_heads)
    avg_iteration_inv = np.mean(iteration_inv, axis=2)       # (n_layers, n_heads)

    # Heads that satisfy invariance condition
    is_iteration_head = avg_iteration_inv > inv_threshold

    # Optional: EoI locator metric (not used for selection, but useful for inspection)
    eoi_peaky = scores["attn_eoi_peaky_thres"]
    avg_eoi_peaky = np.mean(eoi_peaky, axis=2)
    is_eoi_locator = avg_eoi_peaky > peaky_threshold

    print(
        f"\nIteration Head Identification (thresholds: "
        f"peaky>{peaky_threshold}, inv>{inv_threshold}):"
    )
    print("-" * 60)

    iteration_heads_found = []
    n_layers_total, n_heads_total = avg_iteration_peaky.shape
    for layer_idx in range(n_layers_total):
        for head_idx in range(n_heads_total):
            peaky_score = avg_iteration_peaky[layer_idx, head_idx]
            inv_score = avg_iteration_inv[layer_idx, head_idx]
            eoi_score = avg_eoi_peaky[layer_idx, head_idx]

            if is_iteration_head[layer_idx, head_idx]:
                iteration_heads_found.append(
                    {
                        "layer": layer_idx,
                        "head": head_idx,
                        "iteration_peaky": float(peaky_score),
                        "iteration_inv": float(inv_score),
                        "eoi_peaky": float(eoi_score),
                        "is_eoi_locator": bool(is_eoi_locator[layer_idx, head_idx]),
                    }
                )

    if not iteration_heads_found:
        print("No heads found with invariance above threshold.")
        print("\nTop 5 candidates (by iteration peaky score):")
        flat_scores = []
        for layer_idx in range(n_layers_total):
            for head_idx in range(n_heads_total):
                flat_scores.append(
                    {
                        "layer": layer_idx,
                        "head": head_idx,
                        "score": avg_iteration_peaky[layer_idx, head_idx],
                        "inv": avg_iteration_inv[layer_idx, head_idx],
                    }
                )
        flat_scores.sort(key=lambda x: x["score"], reverse=True)
        for i, candidate in enumerate(flat_scores[:5]):
            print(
                f"  {i+1}. Layer {candidate['layer']:2d}, Head {candidate['head']:2d}: "
                f"peaky={candidate['score']:.4f}, inv={candidate['inv']:.4f}"
            )
        # Return empty array if nothing passes invariance threshold
        return np.zeros((0, 3), dtype=float)

    # Sort selected heads by iteration_peaky (descending)
    iteration_heads_sorted = sorted(
        iteration_heads_found,
        key=lambda x: x["iteration_peaky"],
        reverse=True,
    )

    print("\nHeads with inv > {:.2f}, sorted by iteration peaky:".format(inv_threshold))
    for h in iteration_heads_sorted:
        print(
            f"  Layer {h['layer']:2d}, Head {h['head']:2d}: "
            f"peaky={h['iteration_peaky']:.4f}, inv={h['iteration_inv']:.4f}, "
            f"EoI_peaky={h['eoi_peaky']:.4f}"
        )

    # Build return array: [layer_idx, head_idx, iteration_peaky]
    head_triplets = np.array(
        [[h["layer"], h["head"], h["iteration_peaky"]] for h in iteration_heads_sorted],
        dtype=float,
    )

    print(f"\nTotal iteration-like heads (inv>{inv_threshold}): {len(head_triplets)}")
    return head_triplets


