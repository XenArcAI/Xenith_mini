import jax
import jax.numpy as jnp
from jax import jit, pmap, lax
from jax.tree_util import tree_map
import haiku as hk
from transformers import AutoTokenizer
import os
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass, field
from functools import partial
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MISTRAL_VOCAB_SIZE = 32000
HF_TOKEN = "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"

@dataclass
class ModelArgs:
    max_batch_size: int = 64
    max_seq_len: int = 262144
    vocab_size: int = MISTRAL_VOCAB_SIZE  # Will be automatched to tokenizer
    dim: int = 8192
    inter_dim: int = 24576
    moe_inter_dim: int = 3072
    n_layers: int = 64
    n_dense_layers: int = 8
    n_heads: int = 64
    n_kv_heads: int = 16
    n_routed_experts: int = 32
    n_activated_experts: int = 6
    head_dim: int = 128
    rope_theta: float = 500000.0
    chunk_size: int = 8192
    window_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    global_tokens: int = 256
    stm_size: int = 16384
    ltm_max_chunks: int = 32768
    summarizer_n_layers: int = 4
    summarizer_n_heads: int = 16
    lora_rank: int = 128
    norm_eps: float = 1e-5

# Utility Functions
def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    return jnp.cos(freqs) + 1j * jnp.sin(freqs)

def apply_rotary_emb(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    xq_c = xq.reshape(*xq.shape[:-1], -1, 2).view(jnp.complex64)
    xk_c = xk.reshape(*xk.shape[:-1], -1, 2).view(jnp.complex64)
    freqs_cis = freqs_cis[:xq.shape[1]]
    xq_out = (xq_c * freqs_cis).view(jnp.float32).reshape(*xq.shape)
    xk_out = (xk_c * freqs_cis).view(jnp.float32).reshape(*xk.shape)
    return xq_out, xk_out

# Haiku Modules
class LoRALinear(hk.Module):
    def __init__(self, in_features: int, out_features: int, lora_rank: int, name: str = None):
        super().__init__(name=name)
        self.w = hk.get_parameter("w", [out_features, in_features], init=hk.initializers.VarianceScaling())
        self.lora_A = hk.get_parameter("lora_A", [lora_rank, in_features], init=hk.initializers.VarianceScaling())
        self.lora_B = hk.get_parameter("lora_B", [out_features, lora_rank], init=jnp.zeros)
        self.scaling = 1.0 / lora_rank

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = x @ self.w.T
        y += ((x @ self.lora_A.T) @ self.lora_B.T) * self.scaling
        return y

class RMSNorm(hk.Module):
    def __init__(self, dim: int, eps: float = 1e-5, name: str = None):
        super().__init__(name=name)
        self.eps = eps
        self.weight = hk.get_parameter("weight", [dim], init=jnp.ones)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        return x * lax.rsqrt(variance + self.eps) * self.weight

class SparseAttention(hk.Module):
    def __init__(self, args: ModelArgs, name: str = None):
        super().__init__(name=name)
        self.args = args
        self.wq = LoRALinear(args.dim, args.n_heads * args.head_dim, args.lora_rank, name="wq")
        self.wk = LoRALinear(args.dim, args.n_kv_heads * args.head_dim, args.lora_rank, name="wk")
        self.wv = LoRALinear(args.dim, args.n_kv_heads * args.head_dim, args.lora_rank, name="wv")
        self.wo = LoRALinear(args.n_heads * args.head_dim, args.dim, args.lora_rank, name="wo")
        self.scale = args.head_dim ** -0.5
        self.n_rep = args.n_heads // args.n_kv_heads
        self.scale_weights = hk.get_parameter("scale_weights", [len(args.window_sizes)], init=jnp.ones)

    def _repeat_kv(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.repeat(x, self.n_rep, axis=2) if self.n_rep > 1 else x

    def _make_sparse_mask(self, q_len: int, k_len: int) -> jnp.ndarray:
        masks = jnp.zeros((len(self.args.window_sizes), q_len, k_len))
        arange_q = jnp.arange(q_len)
        arange_k = jnp.arange(k_len)
        for i, w in enumerate(self.args.window_sizes):
            masks = masks.at[i].set((jnp.abs(arange_k[None, :] - arange_q[:, None]) <= w // 2).astype(jnp.float32))
            masks = masks.at[i, :, :self.args.global_tokens].set(1.0)
        return masks

    def __call__(self, x: jnp.ndarray, freqs_cis: jnp.ndarray, start_pos: int, kv_cache: Optional[Tuple] = None) -> Tuple[jnp.ndarray, Tuple]:
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).reshape(bsz, seqlen, self.args.n_heads, self.args.head_dim)
        xk = self.wk(x).reshape(bsz, seqlen, self.args.n_kv_heads, self.args.head_dim)
        xv = self.wv(x).reshape(bsz, seqlen, self.args.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[start_pos:start_pos + seqlen])

        if kv_cache is None:
            k_cache, v_cache = xk, xv
        else:
            k_cache, v_cache = kv_cache
            k_cache = jnp.concatenate([k_cache, xk], axis=1)[:, -self.args.max_seq_len:]
            v_cache = jnp.concatenate([v_cache, xv], axis=1)[:, -self.args.max_seq_len:]

        key, value = map(self._repeat_kv, (k_cache, v_cache))
        scores = jax.lax.batch_matmul(xq, key.swapaxes(-1, -2)) * self.scale
        masks = self._make_sparse_mask(seqlen, key.shape[1])
        multi_scores = scores[:, None, :, :, :] - (1.0 - masks[None, :, :, :]) * 1e4
        multi_attn = jax.nn.softmax(multi_scores, axis=-1)
        weights = jax.nn.softmax(self.scale_weights)[None, :, None, None, None]
        attn_weights = (multi_attn * weights).sum(axis=1)
        out = jax.lax.batch_matmul(attn_weights, value).reshape(bsz, seqlen, -1)
        return self.wo(out), (k_cache, v_cache)

class MLP(hk.Module):
    def __init__(self, dim: int, inter_dim: int, args: ModelArgs, name: str = None):
        super().__init__(name=name)
        self.w1 = LoRALinear(dim, inter_dim, args.lora_rank, name="w1")
        self.w3 = LoRALinear(dim, inter_dim, args.lora_rank, name="w3")
        self.w2 = LoRALinear(inter_dim, dim, args.lora_rank, name="w2")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))

class MoE(hk.Module):
    def __init__(self, args: ModelArgs, name: str = None):
        super().__init__(name=name)
        self.args = args
        self.scorer = LoRALinear(args.dim, args.n_routed_experts, args.lora_rank // 2, name="scorer")
        self.experts = [MLP(args.dim, args.moe_inter_dim, args, name=f"expert_{i}") for i in range(args.n_routed_experts)]
        self.shared = MLP(args.dim, args.moe_inter_dim, args, name="shared")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scores = jax.nn.softmax(self.scorer(x.reshape(-1, self.args.dim)), axis=-1)
        top_k_weights, top_k_indices = jax.lax.top_k(scores, self.args.n_activated_experts)
        top_k_weights /= top_k_weights.sum(axis=-1, keepdims=True)

        expert_outputs = jnp.zeros((x.shape[0] * x.shape[1], self.args.dim))
        for i in range(self.args.n_routed_experts):
            mask = (top_k_indices == i).any(axis=-1)
            if mask.any():
                expert_outputs = expert_outputs.at[mask].set(self.experts[i](x.reshape(-1, self.args.dim)[mask]))
        y = (expert_outputs.reshape(*x.shape) * top_k_weights[..., None]).sum(axis=-2)
        return y + self.shared(x)

class TransformerSummarizer(hk.Module):
    def __init__(self, args: ModelArgs, name: str = None):
        super().__init__(name=name)
        self.encoder = hk.nets.Transformer(
            num_heads=args.summarizer_n_heads,
            num_layers=args.summarizer_n_layers,
            d_model=args.dim,
            d_ff=args.moe_inter_dim,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
        )
        self.projector = LoRALinear(args.dim, args.dim, args.lora_rank, name="projector")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.projector(self.encoder(x).mean(axis=1)).reshape(-1, 1, self.projector.w.shape[1])

class XenArcAIi2Beast(hk.Module):
    def __init__(self, args: ModelArgs, name: str = None):
        super().__init__(name=name)
        self.args = args
        self.token_embedding = hk.Embed(args.vocab_size, args.dim)
        self.layers = [
            (SparseAttention(args, name=f"attn_{i}"), MLP(args.dim, args.inter_dim, args, name=f"mlp_{i}") if i < args.n_dense_layers else MoE(args, name=f"moe_{i}"))
            for i in range(args.n_layers)
        ]
        self.norm = RMSNorm(args.dim, args.norm_eps, name="norm")
        self.output_layer = LoRALinear(args.dim, args.vocab_size, args.lora_rank, name="output")
        self.summarizer = TransformerSummarizer(args, name="summarizer")
        self.freqs_cis = precompute_freqs_cis(args.head_dim, args.max_seq_len, args.rope_theta)

    def __call__(self, tokens: jnp.ndarray, start_pos: int = 0, kv_cache: Optional[List[Tuple]] = None, memory: Optional[Dict] = None) -> Tuple[jnp.ndarray, List[Tuple], Dict]:
        if tokens.shape[1] > self.args.max_seq_len:
            tokens = tokens[:, -self.args.max_seq_len:]
            start_pos = max(0, start_pos - (tokens.shape[1] - self.args.max_seq_len))
        h = self.token_embedding(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos + h.shape[1]]
        new_kv_cache = []
        kv_cache = kv_cache or [None] * len(self.layers)
        memory = memory or {"stm": jnp.zeros((1, self.args.stm_size, self.args.dim)), "ltm": {}, "pos": 0}

        # Update STM and LTM
        seq_len = h.shape[1]
        start = memory["pos"] % self.args.stm_size
        end = min(start + seq_len, self.args.stm_size)
        stm = memory["stm"].at[:, start:end].set(h[:, :end - start])
        if seq_len > self.args.stm_size - start:
            stm = stm.at[:, :seq_len - (self.args.stm_size - start)].set(h[:, self.args.stm_size - start:])
        pos = memory["pos"] + seq_len
        if pos >= self.args.stm_size:
            summary = self.summarizer(stm)
            ltm = memory["ltm"]
            ltm[hash(str(tokens[0]))] = summary[0]
            if len(ltm) > self.args.ltm_max_chunks:
                ltm.pop(next(iter(ltm)))
            stm = jnp.zeros_like(stm)
            pos = 0
        memory = {"stm": stm, "ltm": ltm, "pos": pos}

        # Forward pass
        for (attn, ffn), cache in zip(self.layers, kv_cache):
            h, new_cache = attn(h, freqs_cis, start_pos, cache)
            h = h + ffn(h)
            new_kv_cache.append(new_cache)
        h = self.norm(h)
        logits = self.output_layer(h)
        return logits, new_kv_cache, memory

# JAX-transformed model with JIT and pmap
def create_model_fn(args: ModelArgs):
    @hk.transform
    def model_fn(tokens, start_pos=0, kv_cache=None, memory=None):
        model = XenArcAIi2Beast(args)
        return model(tokens, start_pos, kv_cache, memory)
    return model_fn

def parallel_forward(params, tokens, start_pos=0, kv_cache=None, memory=None, rng=None):
    n_devices = jax.local_device_count()
    tokens = tokens.reshape(n_devices, -1, *tokens.shape[1:])
    if kv_cache:
        kv_cache = tree_map(lambda x: x.reshape(n_devices, -1, *x.shape[1:]), kv_cache)
    if memory:
        memory["stm"] = memory["stm"].reshape(n_devices, -1, *memory["stm"].shape[1:])
    model_fn = create_model_fn(args)
    pmap_fn = pmap(jit(lambda p, t, s, c, m: model_fn.apply(p, rng, t, s, c, m)), in_axes=(None, 0, None, 0, None))
    logits, new_kv_cache, new_memory = pmap_fn(params, tokens, start_pos, kv_cache, memory)
    return (logits.reshape(-1, *logits.shape[2:]),
            tree_map(lambda x: x.reshape(-1, *x.shape[2:]), new_kv_cache),
            {"stm": new_memory["stm"].reshape(-1, *new_memory["stm"].shape[2:]), "ltm": new_memory["ltm"], "pos": new_memory["pos"]})

if __name__ == "__main__":
    # Download and save Mistral 8x7B tokenizer
    model_name = "mistralai/Mixtral-8x7B-v0.1"
    logger.info(f"Downloading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    
    # Verify and automatch vocab size
    tokenizer_vocab_size = tokenizer.vocab_size
    logger.info(f"Mixtral-8x7B tokenizer vocab size: {tokenizer_vocab_size}")
    if tokenizer_vocab_size != MISTRAL_VOCAB_SIZE:
        logger.warning(f"Vocab size mismatch! Expected {MISTRAL_VOCAB_SIZE}, got {tokenizer_vocab_size}. Updating ModelArgs.")
        MISTRAL_VOCAB_SIZE = tokenizer_vocab_size  # Automatch
    args = ModelArgs(vocab_size=MISTRAL_VOCAB_SIZE)  # Update args with matched vocab size

    # Save tokenizer as xenith_tokenizer
    save_dir = "./xenith_tokenizer"
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Tokenizer saved to {save_dir}")

    # Load tokenizer (for demo purposes, typically done in training script)
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)
    logger.info("Loaded xenith_tokenizer successfully")

    # Test inference with model
    rng = jax.random.PRNGKey(0)
    model_fn = create_model_fn(args)
    
    # Initialize model
    sample_text = "Hello, world!"
    sample_tokens = jnp.array(loaded_tokenizer.encode(sample_text, add_special_tokens=True), dtype=jnp.int32)[None, :]
    sample_tokens = jnp.pad(sample_tokens, ((0, 0), (0, 2048 - sample_tokens.shape[1])), constant_values=loaded_tokenizer.pad_token_id)
    params = model_fn.init(rng, sample_tokens)
    
    # Run inference
    logger.info(f"Detected {jax.local_device_count()} TPU cores.")
    logits, kv_cache, memory = parallel_forward(params, sample_tokens, rng=rng)
    logger.info(f"Logits shape: {logits.shape}")