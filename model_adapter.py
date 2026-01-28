"""
适配器：将自定义 LlamaForCausalLM 或 GPTNeoXForCausalLM 模型包装成兼容 TransformerLens API 的格式

这个适配器允许你在 head_detect_llama_jx.py 中使用自定义模型，
而不需要 TransformerLens 的 HookedTransformer。

支持两种模型类型：
- LlamaForCausalLM (需要自定义实现)
- GPTNeoXForCausalLM (标准 HuggingFace 实现)
"""

import torch
from typing import Union, List, Optional
from transformers import AutoTokenizer, GPTNeoXForCausalLM
import sys
from pathlib import Path

# 添加路径以导入自定义模型和 hook_utils
# _current_dir = Path(__file__).parent
# _retrieval_head_dir = _current_dir.parent.parent / "Retrieval_Head"
# sys.path.insert(0, str(_retrieval_head_dir / "faiss_attn"))
# sys.path.insert(0, str(_retrieval_head_dir))
import sys
sys.path.append("./faiss_attn/")

try:
    from source.modeling_llama import LlamaForCausalLM as CustomLlamaForCausalLM
    HAS_CUSTOM_LLAMA = True
except ImportError:
    HAS_CUSTOM_LLAMA = False
    CustomLlamaForCausalLM = None

# 导入 hook_utils
try:
    from hook_utils import run_with_cache, ActivationCache
except ImportError as e:
    raise ImportError(
        f"无法导入 hook_utils。请确保路径正确。\n"
    )


class ModelConfig:
    """模拟 TransformerLens 的 cfg 对象"""
    def __init__(self, model):
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.n_ctx = model.config.max_position_embeddings
        # 获取设备
        if hasattr(model, 'device'):
            self.device = model.device
        elif next(model.parameters()).is_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"


class CustomModelAdapter:
    """
    将自定义 LlamaForCausalLM 或 GPTNeoXForCausalLM 模型适配成兼容 TransformerLens API 的格式
    
    使用示例:
        from model_adapter import CustomModelAdapter
        
        # 加载 LLaMA 模型
        model = CustomModelAdapter.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            device="cuda",
            torch_dtype=torch.bfloat16
        )
        
        # 或加载 GPTNeoX/Pythia 模型
        model = CustomModelAdapter.from_pretrained(
            "EleutherAI/pythia-6.9b-deduped",
            device="cuda",
            torch_dtype=torch.bfloat16,
            revision="step3000"
        )
        
        # 现在可以像使用 HookedTransformer 一样使用
        tokens = model.to_tokens("Hello world")
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        attn_pattern = cache["pattern", 0, "attn"]
    """
    
    def __init__(self, model: Union[CustomLlamaForCausalLM, GPTNeoXForCausalLM], tokenizer=None):
        """
        Args:
            model: LlamaForCausalLM 或 GPTNeoXForCausalLM 模型
            tokenizer: 可选的 tokenizer（如果不提供，会从模型名称自动加载）
        """
        self._model = model
        self._model.eval()  # 确保是 eval 模式
        
        # 创建配置对象
        self.cfg = ModelConfig(model)
        
        # 加载 tokenizer
        if tokenizer is None:
            # 尝试从模型配置中获取 tokenizer 名称
            model_name = getattr(model.config, '_name_or_path', None)
            if model_name is None:
                # 尝试从模型路径推断
                if hasattr(model, 'name_or_path'):
                    model_name = model.name_or_path
                else:
                    raise ValueError(
                        "无法自动推断 tokenizer 名称。请提供 tokenizer 参数。"
                    )
            
            # 对于 Pythia 模型，使用 GPTNeoX tokenizer（Pythia 使用相同的 tokenizer）
            is_pythia = "pythia" in model_name.lower()
            
            if is_pythia:
                # Pythia 模型使用 GPTNeoX tokenizer
                # 使用基础 tokenizer 模型，因为 Pythia 的 tokenizer 文件可能有问题
                tokenizer_model = "EleutherAI/gpt-neox-20b"
                print(f"📦 Detected Pythia model, using GPTNeoX tokenizer from: {tokenizer_model}")
                try:
                    # 尝试使用 fast tokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to load fast tokenizer: {e}")
                    try:
                        # 回退到 slow tokenizer
                        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=False)
                    except Exception as e2:
                        print(f"⚠️  Warning: Failed to load slow tokenizer: {e2}")
                        # 最后尝试直接从模型加载，但使用 use_fast=False
                        print(f"   Trying to load tokenizer directly from model...")
                        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
            else:
                # 非 Pythia 模型，使用原来的逻辑
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self._tokenizer = tokenizer
        
        # 确保 tokenizer 有 pad_token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # 暴露 tokenizer 作为公共属性（用于兼容性，如 iteration_head_detector）
        self.tokenizer = self._tokenizer
    
    def to_tokens(self, seq: Union[str, List[str]], prepend_bos: bool = True) -> torch.Tensor:
        """
        将文本转换为 token IDs，兼容 TransformerLens 的 to_tokens 方法
        
        Args:
            seq: 字符串或字符串列表
            prepend_bos: 是否在开头添加 BOS token（TransformerLens 默认行为）
        
        Returns:
            Token IDs tensor，shape: [batch_size, seq_len]
        """
        if isinstance(seq, str):
            seq = [seq]
        
        # Tokenize
        encoded = self._tokenizer(
            seq,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.n_ctx
        )
        
        input_ids = encoded['input_ids']
        
        # TransformerLens 默认会 prepend_bos，但我们的 tokenizer 可能已经包含了
        # 检查第一个 token 是否是 BOS
        if prepend_bos and self._tokenizer.bos_token_id is not None:
            # 检查是否已经有 BOS token
            has_bos = (input_ids[:, 0] == self._tokenizer.bos_token_id).all()
            if not has_bos:
                # 添加 BOS token
                bos_tensor = torch.full(
                    (input_ids.shape[0], 1),
                    self._tokenizer.bos_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device
                )
                input_ids = torch.cat([bos_tensor, input_ids], dim=1)
        
        # 移动到模型设备
        if hasattr(self._model, 'device'):
            input_ids = input_ids.to(self._model.device)
        elif next(self._model.parameters()).is_cuda:
            input_ids = input_ids.cuda()
        
        return input_ids
    
    def to_str_tokens(self, seq: Union[str, List[str], torch.Tensor], prepend_bos: bool = True) -> List[str]:
        """
        将文本或 token IDs 转换为 token 字符串列表，兼容 TransformerLens 的 to_str_tokens 方法
        
        Args:
            seq: 字符串、字符串列表或 token IDs tensor
            prepend_bos: 是否在开头添加 BOS token
        
        Returns:
            Token 字符串列表
        """
        if isinstance(seq, torch.Tensor):
            # 如果是 tensor，直接解码
            tokens = seq.cpu().tolist()
            if isinstance(tokens[0], list):
                # batch 情况，取第一个
                tokens = tokens[0]
            return self._tokenizer.convert_ids_to_tokens(tokens)
        elif isinstance(seq, str):
            # 如果是字符串，先 tokenize 再转换
            input_ids = self.to_tokens(seq, prepend_bos=prepend_bos)
            return self.to_str_tokens(input_ids, prepend_bos=False)
        else:
            # 字符串列表
            result = []
            for s in seq:
                tokens = self.to_str_tokens(s, prepend_bos=prepend_bos)
                result.extend(tokens)
            return result
    
    def run_with_cache(
        self,
        tokens: torch.Tensor,
        remove_batch_dim: bool = True
    ) -> tuple:
        """
        运行模型并获取 cache，兼容 TransformerLens 的 run_with_cache 方法
        
        Args:
            tokens: Token IDs tensor
            remove_batch_dim: 是否移除 batch 维度（当 batch_size=1 时）
        
        Returns:
            Tuple of (logits, cache)
            注意：返回的 cache 是 ActivationCache 对象，支持 cache["pattern", layer, "attn"] 访问
        """
        logits, cache, attentions = run_with_cache(
            self._model,
            tokens,
            remove_batch_dim=remove_batch_dim,
            output_attentions=True,
            attn_mode="torch"  # 使用 torch 模式获取 attention weights
        )
        
        # 返回 (logits, cache)，与 TransformerLens 保持一致
        return logits, cache
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = None,
        model_type: Optional[str] = None,  # "llama" or "gptneox" or None (auto-detect)
        **kwargs
    ):
        """
        从预训练模型加载并创建适配器，兼容 TransformerLens 的 from_pretrained 方法
        
        Args:
            model_name: 模型名称或路径
            device: 设备 ("cuda" 或 "cpu")
            torch_dtype: 数据类型（如 torch.bfloat16, torch.float16）
            model_type: 模型类型 ("llama" 或 "gptneox")，如果为 None 则自动检测
            **kwargs: 传递给模型 from_pretrained 的其他参数
        
        Returns:
            CustomModelAdapter 实例
        """
        # 自动检测模型类型
        if model_type is None:
            model_name_lower = model_name.lower()
            if "pythia" in model_name_lower or "gpt-neox" in model_name_lower or "gptneox" in model_name_lower:
                model_type = "gptneox"
            elif "llama" in model_name_lower:
                model_type = "llama"
            else:
                # 尝试根据模型配置判断
                try:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(model_name, **{k: v for k, v in kwargs.items() if k in ['revision', 'trust_remote_code']})
                    if config.model_type == "gpt_neox":
                        model_type = "gptneox"
                    elif config.model_type == "llama":
                        model_type = "llama"
                    else:
                        # 默认尝试 LLaMA（如果自定义模型可用）
                        if HAS_CUSTOM_LLAMA:
                            model_type = "llama"
                        else:
                            model_type = "gptneox"
                except:
                    # 如果无法检测，默认使用 GPTNeoX（标准 HuggingFace）
                    model_type = "gptneox"
        
        # 默认参数
        default_kwargs = {
            'torch_dtype': torch_dtype or (torch.bfloat16 if device == "cuda" else torch.float32),
            'device_map': 'auto' if device == "cuda" else None,
        }
        
        # 根据模型类型设置特定参数
        if model_type == "llama":
            if not HAS_CUSTOM_LLAMA:
                raise ImportError(
                    "LLaMA 模型需要自定义实现，但无法导入 CustomLlamaForCausalLM。"
                    "请确保 source.modeling_llama 模块可用。"
                )
            # LLaMA 特定参数
            default_kwargs['use_flash_attention_2'] = kwargs.get('use_flash_attention_2', "flash_attention_2")
            if 'use_flash_attention_2' in kwargs:
                default_kwargs['use_flash_attention_2'] = kwargs['use_flash_attention_2']
        
        default_kwargs.update(kwargs)
        
        # 加载模型
        if model_type == "llama":
            model = CustomLlamaForCausalLM.from_pretrained(
                model_name,
                **default_kwargs
            )
            
            # ⚠️ LLaMA 特定修复: Flash Attention 2 处理
            if hasattr(model.config, '_attn_implementation'):
                if model.config._attn_implementation == "eager":
                    import warnings
                    warnings.warn(
                        "Model was loaded with eager attention implementation, but custom model code requires "
                        "LlamaFlashAttention2 class (which has forward_torch method). "
                        "Forcing use of flash_attention_2 implementation. "
                        "You can still control whether to use Flash Attention via attn_mode parameter."
                    )
                    model.config._attn_implementation = "flash_attention_2"
                    
                    try:
                        from source.modeling_llama import LLAMA_ATTENTION_CLASSES
                    except ImportError:
                        import warnings
                        warnings.warn(
                            "无法导入 LLAMA_ATTENTION_CLASSES，跳过 attention 模块修复。"
                            "模型可能无法正常工作。"
                        )
                        model.eval()
                        return cls(model)
                    
                    model_device = next(model.parameters()).device
                    model_dtype = next(model.parameters()).dtype
                    
                    for layer_idx, layer in enumerate(model.model.layers):
                        if hasattr(layer, 'self_attn'):
                            old_attn = layer.self_attn
                            old_device = next(old_attn.parameters()).device
                            old_dtype = next(old_attn.parameters()).dtype
                            
                            new_attn = LLAMA_ATTENTION_CLASSES["flash_attention_2"](
                                config=model.config,
                                layer_idx=layer_idx
                            )
                            new_attn.load_state_dict(old_attn.state_dict())
                            new_attn = new_attn.to(device=old_device, dtype=old_dtype)
                            layer.self_attn = new_attn
        else:  # GPTNeoX
            # GPTNeoX 使用标准 HuggingFace 实现，不需要特殊处理
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                **default_kwargs
            )
        
        # 如果不是自动设备映射，手动移动到设备
        if default_kwargs.get('device_map') is None:
            model = model.to(device)
        
        model.eval()
        
        # 创建适配器
        return cls(model)
    
    def __call__(self, *args, **kwargs):
        """直接调用底层模型"""
        return self._model(*args, **kwargs)
    
    def __getattr__(self, name):
        """转发其他属性访问到底层模型"""
        # tokenizer 已经在 __init__ 中设置为公共属性 self.tokenizer
        # 如果访问 tokenizer，应该直接返回（不应该到达这里）
        # 但如果确实到达这里，说明 tokenizer 属性可能不存在，尝试返回它
        if name == "tokenizer":
            # 如果 self.tokenizer 不存在，尝试返回 _tokenizer
            if hasattr(self, '_tokenizer'):
                return self._tokenizer
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'tokenizer'")
        # 其他属性转发到底层模型
        return getattr(self._model, name)


# 为了兼容性，也可以创建一个别名
HookedTransformer = CustomModelAdapter

