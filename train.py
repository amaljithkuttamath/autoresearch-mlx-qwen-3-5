"""
Autoresearch pretraining script. Single-device, single-file.
Tiny Qwen3.5-style hybrid architecture (Gated DeltaNet + Full Attention) on MLX.
Usage: uv run train.py
"""

import gc
import math
import os
import sys
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

try:
    import wandb
    USE_WANDB = "--wandb" in sys.argv  # opt-in: pass --wandb to enable
    if USE_WANDB:
        os.environ.setdefault("WANDB_MODE", "offline")
except ImportError:
    wandb = None
    USE_WANDB = False

VERBOSE = "--verbose" in sys.argv
VERBOSE_INTERVAL = 50  # log tensor stats every N steps (always log on NaN)
_verbose_this_step = False  # set per-step

# ---------------------------------------------------------------------------
# Logging: writes to logs/<timestamp>.jsonl when --verbose
# ---------------------------------------------------------------------------
import json
from datetime import datetime

_log_file = None
_log_step = 0

if VERBOSE:
    os.makedirs("logs", exist_ok=True)
    _run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = f"logs/{_run_id}.jsonl"
    _log_file = open(_log_path, "w")
    print(f"Verbose logging to {_log_path}")


def _log_event(event_type, data):
    """Append a JSON line to the log file."""
    if _log_file is None:
        return
    record = {"step": _log_step, "type": event_type, **data}
    _log_file.write(json.dumps(record) + "\n")
    _log_file.flush()


def _ts(name, x):
    """Tensor stats: name, shape, min/max/mean/has_nan. Only runs on verbose steps."""
    if not _verbose_this_step:
        return
    x_eval = mx.array(x)
    mx.eval(x_eval)
    flat = x_eval.reshape(-1).astype(mx.float32)
    has_nan = bool(mx.any(mx.isnan(flat)).item())
    has_inf = bool(mx.any(mx.isinf(flat)).item())
    mn = float(flat.min().item()) if not has_nan else float("nan")
    mx_val = float(flat.max().item()) if not has_nan else float("nan")
    mean = float(mx.mean(flat).item()) if not has_nan else float("nan")
    flag = ""
    if has_nan:
        flag = " *** NaN ***"
    elif has_inf:
        flag = " *** Inf ***"
    print(f"  [{name}] shape={list(x.shape)} min={mn:.4g} max={mx_val:.4g} mean={mean:.4g}{flag}")
    _log_event("tensor", {
        "name": name, "shape": list(x.shape),
        "min": mn, "max": mx_val, "mean": mean,
        "has_nan": has_nan, "has_inf": has_inf,
    })


def _scalar(name, val):
    """Log a scalar value. Only runs on verbose steps."""
    if not _verbose_this_step:
        return
    print(f"  [{name}] = {val}")
    _log_event("scalar", {"name": name, "value": val})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Qwen35Config:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 256
    mlp_expansion: float = 3.5
    head_dim: int = 64
    # DeltaNet config
    deltanet_heads: int = 4
    deltanet_head_dim: int = 64
    conv_kernel_size: int = 4
    # Layer pattern: "L" = linear (DeltaNet), "F" = full attention
    layer_pattern: str = "LLLF"


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Qwen3.5-style RMSNorm with (1 + weight) scaling, zero-init."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dim,))

    def __call__(self, x):
        x_f32 = x.astype(mx.float32)
        variance = mx.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        x_norm = x_f32 * mx.rsqrt(variance + self.eps)
        return (x_norm * (1.0 + self.weight.astype(mx.float32))).astype(x.dtype)


class RMSNormGated(nn.Module):
    """Qwen3.5-style gated RMSNorm for DeltaNet output."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x, gate):
        x_f32 = x.astype(mx.float32)
        variance = mx.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        x_norm = x_f32 * mx.rsqrt(variance + self.eps)
        x_norm = self.weight.astype(mx.float32) * x_norm
        return (x_norm * nn.silu(gate.astype(mx.float32))).astype(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x))"""
    def __init__(self, config):
        super().__init__()
        hidden = int(config.n_embd * config.mlp_expansion)
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def l2norm(x, eps=1e-6):
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


def create_additive_causal_mask(seq_len, dtype=mx.bfloat16):
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def get_peak_memory_mb():
    return mx.get_peak_memory() / 1024 / 1024


# ---------------------------------------------------------------------------
# Full Attention (Qwen3.5-style with gated Q)
# ---------------------------------------------------------------------------

class FullAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.group_size = self.n_head // self.n_kv_head

        # Gated Q: projects to 2x dim, split into Q and gate
        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim * 2, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x, mask):
        B, T, _ = x.shape

        # Q with gate
        q_and_gate = self.c_q(x).reshape(B, T, self.n_head, self.head_dim * 2)
        q, gate = mx.split(q_and_gate, 2, axis=-1)
        gate = gate.reshape(B, T, self.n_head * self.head_dim)

        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (B, heads, T, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Expand KV heads for GQA
        if self.group_size > 1:
            k = mx.repeat(k, self.group_size, axis=1)
            v = mx.repeat(v, self.group_size, axis=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)

        # Apply sigmoid gate
        y = y * mx.sigmoid(gate)
        return self.c_proj(y)


# ---------------------------------------------------------------------------
# Gated DeltaNet (recurrent, step-by-step)
# ---------------------------------------------------------------------------

class GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_heads = config.deltanet_heads
        self.head_dim = config.deltanet_head_dim
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.num_heads * self.head_dim
        self.conv_kernel_size = config.conv_kernel_size

        # QKV projection + causal conv1d
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(config.n_embd, self.conv_dim, bias=False)

        # Depthwise causal conv1d weights: (conv_dim, 1, kernel_size)
        self.conv_weight = mx.zeros((self.conv_dim, 1, self.conv_kernel_size))

        # Decay (A_log) and timestep bias
        self.A_log = mx.zeros((self.num_heads,))
        self.dt_bias = mx.ones((self.num_heads,))

        # Beta (write strength) and alpha (decay gate) projections
        self.in_proj_b = nn.Linear(config.n_embd, self.num_heads, bias=False)
        self.in_proj_a = nn.Linear(config.n_embd, self.num_heads, bias=False)

        # Output gate
        self.in_proj_z = nn.Linear(config.n_embd, self.value_dim, bias=False)

        # Output norm and projection
        self.norm = RMSNormGated(self.head_dim)
        self.out_proj = nn.Linear(self.value_dim, config.n_embd, bias=False)

    def _causal_conv1d(self, x):
        """Apply causal depthwise conv1d + silu. x: (B, conv_dim, T)"""
        B, C, T = x.shape
        # Pad on the left for causal convolution
        x_padded = mx.pad(x, [(0, 0), (0, 0), (self.conv_kernel_size - 1, 0)])
        # Depthwise conv: process each channel independently
        out = mx.zeros((B, C, T))
        for k in range(self.conv_kernel_size):
            # conv_weight[:, 0, k] is (C,), need (1, C, 1) for broadcast
            w = self.conv_weight[:, 0, k].reshape(1, C, 1)
            out = out + x_padded[:, :, self.conv_kernel_size - 1 - k:self.conv_kernel_size - 1 - k + T] * w
        return nn.silu(out)

    @staticmethod
    def _chunk_forward(q, k, v, beta, g, chunk_size=64):
        """
        Chunk-wise gated delta rule, faithful port of FLA naive_chunk_gated_delta_rule.

        Args:
            q, k: (B, T, H, D_k) -- l2-normalized, q already scaled by 1/sqrt(d)
            v:    (B, T, H, D_v)
            beta: (B, T, H)
            g:    (B, T, H) -- log-space decay (negative values)

        Returns:
            output: (B, T, H, D_v)
        """
        B, T, H, D = q.shape
        C = chunk_size

        # Pre-scale v by beta and compute k_beta (FLA convention)
        v = v * beta[:, :, :, None]
        k_beta = k * beta[:, :, :, None]

        # Pad T to multiple of C
        pad_len = (C - T % C) % C
        if pad_len > 0:
            q = mx.pad(q, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
            k = mx.pad(k, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
            v = mx.pad(v, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
            k_beta = mx.pad(k_beta, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
            g = mx.pad(g, [(0, 0), (0, pad_len), (0, 0)])

        T_pad = T + pad_len
        nc = T_pad // C

        # Transpose to (B, H, T, D) then reshape to (B, H, nc, C, D)
        q = q.transpose(0, 2, 1, 3).reshape(B, H, nc, C, D)
        k = k.transpose(0, 2, 1, 3).reshape(B, H, nc, C, D)
        v = v.transpose(0, 2, 1, 3).reshape(B, H, nc, C, D)
        k_beta = k_beta.transpose(0, 2, 1, 3).reshape(B, H, nc, C, D)
        g = g.transpose(0, 2, 1).reshape(B, H, nc, C)  # (B, H, nc, C)

        # Cumulative decay within each chunk
        g_cumsum = mx.cumsum(g, axis=-1)  # (B, H, nc, C)
        decay_exp = mx.exp(g_cumsum)[:, :, :, :, None]  # (B, H, nc, C, 1)

        # Intra-chunk decay mask: L[i,j] = exp(g_cumsum[i] - g_cumsum[j]) for i >= j
        g_i = g_cumsum[:, :, :, :, None]  # (B,H,nc,C,1)
        g_j = g_cumsum[:, :, :, None, :]  # (B,H,nc,1,C)
        # Mask BEFORE exp to avoid inf: upper triangle diffs can be large positive
        tri = mx.tri(C, C, k=0)
        L_diff = (g_i - g_j) * tri  # zero out upper triangle before exp
        L_mask = mx.exp(L_diff) * tri

        # WY matrix (FLA: attn): A = -(k_beta @ k^T) * L_mask, upper triangle zeroed
        A = -(k_beta @ k.transpose(0, 1, 2, 4, 3)) * L_mask
        # Zero the diagonal and upper triangle (FLA uses masked_fill with triu(diagonal=0))
        A = A * mx.tri(C, C, k=-1)

        # Triangular solve (FLA in-place row update, here functional)
        A_rows = [A[:, :, :, 0:1, :]]
        for i in range(1, C):
            row_i = A[:, :, :, i:i+1, :]
            prev = mx.concatenate(A_rows, axis=-2)
            correction = row_i[:, :, :, :, :i] @ prev
            A_rows.append(row_i + correction)
        attn_solved = mx.concatenate(A_rows, axis=-2) + mx.eye(C)

        # FLA pre-computes corrected v and corrected decayed keys for ALL chunks
        # k_cumsum = attn @ v  (WY-corrected values)
        # k_cumdecay = attn @ (k_beta * decay_exp)  (WY-corrected decayed keys)
        v_corrected = attn_solved @ v          # (B, H, nc, C, D)
        k_cumdecay = attn_solved @ (k_beta * decay_exp)  # (B, H, nc, C, D)

        # Upper-triangle mask for intra-chunk attention (zero future positions)
        upper_mask = mx.tri(C, C, k=0)  # lower triangular = 1

        # Per-chunk recurrence
        state = mx.zeros((B, H, D, D))  # (B, H, D_k, D_v)
        outputs = []

        for c in range(nc):
            q_c = q[:, :, c]              # (B, H, C, D)
            k_c = k[:, :, c]
            v_c = v_corrected[:, :, c]     # WY-corrected, beta-scaled values
            kcd_c = k_cumdecay[:, :, c]    # WY-corrected decayed keys

            # Inter-chunk: retrieve from state using corrected decayed keys
            v_prime = kcd_c @ state  # (B,H,C,D) @ (B,H,D,D) -> (B,H,C,D)
            v_new = v_c - v_prime

            # Intra-chunk attention
            g_c = g_cumsum[:, :, c]  # (B, H, C)
            o_intra = (q_c @ k_c.transpose(0, 1, 3, 2) * L_mask[:, :, c]) * upper_mask
            # FLA uses masked_fill with triu(diagonal=1) to zero upper triangle
            # L_mask already has lower-tri structure, so multiply by upper_mask is redundant
            # but we keep it for safety
            o_intra = o_intra @ v_new

            # Inter-chunk: read from state with decay
            q_decayed = q_c * mx.exp(g_c)[:, :, :, None]
            o_inter = q_decayed @ state

            outputs.append(o_intra + o_inter)

            # State update
            g_last = g_cumsum[:, :, c, -1:]  # (B, H, 1)
            decay_state = mx.exp(g_last)[:, :, :, None]  # (B, H, 1, 1)
            g_rel = (g_last[:, :, :, None] - g_c[:, :, :, None])  # (B,H,C,1)
            k_decayed = k_c * mx.exp(g_rel)
            state = state * decay_state + k_decayed.transpose(0, 1, 3, 2) @ v_new

        # Reassemble: (B, H, nc, C, D) -> (B, T, H, D)
        out = mx.stack(outputs, axis=2)
        out = out.reshape(B, H, T_pad, D).transpose(0, 2, 1, 3)  # (B, T_pad, H, D)

        if pad_len > 0:
            out = out[:, :T]

        return out

    def __call__(self, x, _layer_idx=None):
        B, T, _ = x.shape
        _lid = f"DN-L{_layer_idx}" if _layer_idx is not None else "DN"

        # Project to QKV
        mixed = self.in_proj_qkv(x)  # (B, T, conv_dim)
        mixed = mixed.transpose(0, 2, 1)  # (B, conv_dim, T)

        # Causal conv1d + silu
        mixed = self._causal_conv1d(mixed)
        mixed = mixed.transpose(0, 2, 1)  # (B, T, conv_dim)

        # Split into Q, K, V
        q = mixed[:, :, :self.key_dim]
        k = mixed[:, :, self.key_dim:self.key_dim * 2]
        v = mixed[:, :, self.key_dim * 2:]

        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)

        # L2 normalize Q and K
        q = l2norm(q)
        k = l2norm(k)

        # Compute beta (write strength) and g (decay)
        beta = mx.sigmoid(self.in_proj_b(x))  # (B, T, num_heads)
        a = self.in_proj_a(x)  # (B, T, num_heads)
        # Clamp A_log to prevent decay coefficient explosion (init range is [0, 2.77])
        A_log_clamped = mx.clip(self.A_log.astype(mx.float32), 0.0, 4.0)
        A = -mx.exp(A_log_clamped)
        g = A * nn.softplus(a.astype(mx.float32) + self.dt_bias)  # (B, T, num_heads)

        _ts(f"{_lid}/A_log", self.A_log)
        _ts(f"{_lid}/A(decay_coeff)", A)
        _ts(f"{_lid}/dt_bias", self.dt_bias)
        _ts(f"{_lid}/g(log_decay)", g)
        _ts(f"{_lid}/beta", beta)
        _ts(f"{_lid}/v_pre_chunk", v)

        # Output gate
        z = self.in_proj_z(x)  # (B, T, value_dim)

        # Chunk-wise delta rule (FLA convention)
        scale = 1.0 / math.sqrt(self.head_dim)
        q_f = (q * scale).astype(mx.float32)
        k_f = k.astype(mx.float32)
        v_f = v.astype(mx.float32)
        beta_f = beta.astype(mx.float32)
        g_f = g.astype(mx.float32)

        core_out = mx.checkpoint(self._chunk_forward)(q_f, k_f, v_f, beta_f, g_f, chunk_size=64)
        _ts(f"{_lid}/chunk_out", core_out)

        # Reshape for gated norm: norm operates per-head on last dim
        core_out_flat = core_out.reshape(B * T * self.num_heads, self.head_dim)
        z_heads = z.reshape(B, T, self.num_heads, self.head_dim)
        z_flat = z_heads.reshape(B * T * self.num_heads, self.head_dim)
        core_out_flat = self.norm(core_out_flat, z_flat)
        core_out = core_out_flat.reshape(B, T, self.value_dim)
        _ts(f"{_lid}/gated_norm_out", core_out)

        final_out = self.out_proj(core_out)
        _ts(f"{_lid}/final_out", final_out)
        return final_out


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config, layer_idx, layer_type):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        if layer_type == "F":
            self.attn = FullAttention(config)
        else:
            self.attn = GatedDeltaNet(config)
        self.mlp = SwiGLU(config)
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)

    def __call__(self, x, mask):
        if self.layer_type == "F":
            x = x + self.attn(self.norm1(x), mask)
        else:
            x = x + self.attn(self.norm1(x), _layer_idx=self.layer_idx)
        x = x + self.mlp(self.norm2(x))
        _ts(f"block{self.layer_idx}({self.layer_type})/out", x)
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen35(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Parse layer pattern
        pattern = config.layer_pattern.upper()
        assert all(c in "LF" for c in pattern)
        self.layer_types = []
        for i in range(config.n_layer):
            self.layer_types.append(pattern[i % len(pattern)])

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i, self.layer_types[i]) for i in range(config.n_layer)]
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self._mask_cache = {}

    def init_weights(self):
        n_embd = self.config.n_embd
        scale = 3**0.5 * n_embd**-0.5

        self.wte.weight = (mx.random.normal(self.wte.weight.shape) * 1.0).astype(mx.bfloat16)
        self.lm_head.weight = (mx.random.normal(self.lm_head.weight.shape) * 0.001).astype(mx.bfloat16)

        for i, block in enumerate(self.blocks):
            # MLP init
            block.mlp.gate_proj.weight = mx.random.uniform(-scale, scale, block.mlp.gate_proj.weight.shape).astype(mx.bfloat16)
            block.mlp.up_proj.weight = mx.random.uniform(-scale, scale, block.mlp.up_proj.weight.shape).astype(mx.bfloat16)
            block.mlp.down_proj.weight = mx.zeros_like(block.mlp.down_proj.weight).astype(mx.bfloat16)

            # RMSNorm weights are already zero-init (for (1+w) style)
            # block.norm1.weight and block.norm2.weight are zero by default

            if self.layer_types[i] == "F":
                attn = block.attn
                attn.c_q.weight = mx.random.uniform(-scale, scale, attn.c_q.weight.shape).astype(mx.bfloat16)
                attn.c_k.weight = mx.random.uniform(-scale, scale, attn.c_k.weight.shape).astype(mx.bfloat16)
                attn.c_v.weight = mx.random.uniform(-scale, scale, attn.c_v.weight.shape).astype(mx.bfloat16)
                attn.c_proj.weight = mx.zeros_like(attn.c_proj.weight).astype(mx.bfloat16)
            else:
                dn = block.attn
                dn.in_proj_qkv.weight = mx.random.uniform(-scale, scale, dn.in_proj_qkv.weight.shape).astype(mx.bfloat16)
                dn.conv_weight = mx.random.normal(dn.conv_weight.shape).astype(mx.bfloat16) * 0.02
                dn.in_proj_b.weight = mx.random.uniform(-scale, scale, dn.in_proj_b.weight.shape).astype(mx.bfloat16)
                dn.in_proj_a.weight = mx.random.uniform(-scale, scale, dn.in_proj_a.weight.shape).astype(mx.bfloat16)
                dn.in_proj_z.weight = mx.random.uniform(-scale, scale, dn.in_proj_z.weight.shape).astype(mx.bfloat16)
                dn.out_proj.weight = mx.zeros_like(dn.out_proj.weight).astype(mx.bfloat16)
                dn.A_log = mx.log(mx.random.uniform(low=1.0, high=16.0, shape=dn.A_log.shape))
                # Official Qwen3.5: dt_bias = ones
                dn.dt_bias = mx.ones(dn.dt_bias.shape)

    def _get_mask(self, seq_len):
        if seq_len not in self._mask_cache:
            self._mask_cache[seq_len] = create_additive_causal_mask(seq_len)
        return self._mask_cache[seq_len]

    def __call__(self, idx, targets=None, reduction="mean"):
        _, seq_len = idx.shape
        mask = self._get_mask(seq_len)

        x = self.wte(idx)
        _ts("embed_out", x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        _ts("final_norm_out", x)

        logits = self.lm_head(x).astype(mx.float32)
        _ts("logits", logits)

        if targets is None:
            return logits

        valid = targets != -1
        targets_safe = mx.where(valid, targets, mx.zeros_like(targets))
        ce = nn.losses.cross_entropy(logits, targets_safe, reduction="none")
        ce = ce * valid
        _ts("ce_per_token", ce)
        if reduction == "none":
            return ce
        denom = mx.maximum(mx.sum(valid), 1)
        loss = mx.sum(ce) / denom
        _scalar("loss", float(loss.item()) if VERBOSE else "")
        return loss


# ---------------------------------------------------------------------------
# AdamW Optimizer
# ---------------------------------------------------------------------------

class AdamW:
    def __init__(self, model, unembedding_lr, embedding_lr, matrix_lr, weight_decay, adam_betas, scalar_lr, decay_lr):
        self.param_config = {}
        self.adam_state = {}

        model_dim = model.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        flat_params = tree_flatten(model.parameters())
        for path, param in flat_params:
            if "blocks" in path and param.ndim == 2:
                self.param_config[path] = {
                    "lr": matrix_lr,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": weight_decay,
                }
            elif "wte" in path:
                self.param_config[path] = {
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "lm_head" in path:
                self.param_config[path] = {
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "A_log" in path or "dt_bias" in path:
                # Decay params live inside exp() -- need lower LR
                self.param_config[path] = {
                    "lr": decay_lr,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            else:
                # Norm weights, conv_weight, etc.
                self.param_config[path] = {
                    "lr": scalar_lr,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }

        self.initial_lrs = {path: config["lr"] for path, config in self.param_config.items()}

    def _set_path_value(self, model, path, value):
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)

    def _step(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        beta1, beta2 = config["betas"]
        eps = config["eps"]
        weight_decay = config["weight_decay"]

        if path not in self.adam_state:
            self.adam_state[path] = {
                "m": mx.zeros_like(grad_f32),
                "v": mx.zeros_like(grad_f32),
                "t": 0,
            }

        state = self.adam_state[path]
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad_f32
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad_f32 * grad_f32)

        bias1 = 1 - beta1 ** state["t"]
        bias2 = 1 - beta2 ** state["t"]
        denom = mx.sqrt(state["v"] / bias2) + eps
        step_size = lr / bias1

        param_f32 = param_f32 * (1 - lr * weight_decay)
        param_f32 = param_f32 - step_size * (state["m"] / denom)
        return param_f32.astype(param.dtype)

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        for path, grad in flat_grads.items():
            if path not in self.param_config:
                continue
            config = self.param_config[path]
            param = flat_params[path]
            new_param = self._step(path, grad, param, config)
            self._set_path_value(model, path, new_param)

    def set_lr_multiplier(self, multiplier):
        for path, config in self.param_config.items():
            config["lr"] = self.initial_lrs[path] * multiplier

    @property
    def state(self):
        arrays = []
        for state in self.adam_state.values():
            arrays.extend([state["m"], state["v"]])
        return arrays


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Architecture
N_EMBD = 256
N_HEAD = 4
N_KV_HEAD = 4
HEAD_DIM = 64
MLP_EXPANSION = 3.5
LAYER_PATTERN = "LLLF"
DELTANET_HEADS = 4
DELTANET_HEAD_DIM = 64
CONV_KERNEL_SIZE = 4

# Optimizer
TOTAL_BATCH_SIZE = 2**13  # 8192 tokens per optimizer step
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.04
DECAY_LR = 0.04  # A_log, dt_bias -- clamped to [0,4], safe at same LR
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.01
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Training
DEPTH = 4
DEVICE_BATCH_SIZE = 4
FINAL_EVAL_BATCH_SIZE = 4
STARTUP_EXCLUDE_STEPS = 1


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)
t_data = time.time()
print(f"Data/tokenizer loaded in {t_data - t_start:.1f}s")

config = Qwen35Config(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=N_HEAD,
    n_kv_head=N_KV_HEAD,
    n_embd=N_EMBD,
    mlp_expansion=MLP_EXPANSION,
    head_dim=HEAD_DIM,
    deltanet_heads=DELTANET_HEADS,
    deltanet_head_dim=DELTANET_HEAD_DIM,
    conv_kernel_size=CONV_KERNEL_SIZE,
    layer_pattern=LAYER_PATTERN,
)

model = Qwen35(config)
model.init_weights()
mx.eval(model.parameters())
num_params = sum(param.size for _, param in tree_flatten(model.parameters()))

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = AdamW(
    model,
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    adam_betas=ADAM_BETAS,
    scalar_lr=SCALAR_LR,
    decay_lr=DECAY_LR,
)

loss_grad_fn = nn.value_and_grad(model, lambda model, inputs, targets: model(inputs, targets=targets))

if USE_WANDB:
    wandb.init(
        project="autoresearch-mlx",
        config={
            "n_embd": N_EMBD, "n_head": N_HEAD, "head_dim": HEAD_DIM,
            "depth": DEPTH, "layer_pattern": LAYER_PATTERN,
            "total_batch_size": TOTAL_BATCH_SIZE, "seq_len": MAX_SEQ_LEN,
            "matrix_lr": MATRIX_LR, "embedding_lr": EMBEDDING_LR,
            "unembedding_lr": UNEMBEDDING_LR, "scalar_lr": SCALAR_LR,
            "decay_lr": DECAY_LR, "weight_decay": WEIGHT_DECAY,
            "adam_betas": ADAM_BETAS, "warmup_ratio": WARMUP_RATIO,
            "warmdown_ratio": WARMDOWN_RATIO, "time_budget": TIME_BUDGET,
            "num_params_M": num_params / 1e6,
        },
    )

print(f"Architecture: Qwen3.5-style hybrid ({LAYER_PATTERN})")
print(f"Parameters: {num_params / 1e6:.1f}M")
print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

smooth_train_loss = 0.0
total_training_time = 0.0
step = 0
t_compiled = None

# Log run config for reproducibility
_log_event("config", {
    "layer_pattern": LAYER_PATTERN, "n_embd": N_EMBD, "n_head": N_HEAD,
    "head_dim": HEAD_DIM, "depth": DEPTH, "batch_size": TOTAL_BATCH_SIZE,
    "matrix_lr": MATRIX_LR, "scalar_lr": SCALAR_LR, "decay_lr": DECAY_LR, "embedding_lr": EMBEDDING_LR,
    "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
    "num_params_M": num_params / 1e6,
})

_prev_loss_was_nan = False

while True:
    _log_step = step
    # Log on first 3 steps, every VERBOSE_INTERVAL, and the step after a NaN
    _verbose_this_step = VERBOSE and (step < 3 or step % VERBOSE_INTERVAL == 0 or _prev_loss_was_nan)
    t0 = time.time()
    accum_grads = None
    train_loss = None

    if _verbose_this_step:
        print(f"\n{'='*60}\nSTEP {step} FORWARD PASS\n{'='*60}")

    for _ in range(grad_accum_steps):
        loss, grads = loss_grad_fn(model, x, y)
        mx.eval(loss, grads)
        if t_compiled is None:
            t_compiled = time.time()
            print(f"Model compiled in {t_compiled - t_data:.1f}s")
        train_loss = loss
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda lhs, rhs: lhs + rhs, accum_grads, grads)
        x, y, epoch = next(train_loader)

    if grad_accum_steps > 1:
        accum_grads = tree_map(lambda grad: grad * (1.0 / grad_accum_steps), accum_grads)

    # Gradient clipping
    grad_norm_sq = sum(mx.sum(g * g).item() for _, g in tree_flatten(accum_grads))
    grad_norm = grad_norm_sq ** 0.5
    max_grad_norm = 1.0

    # Per-parameter grad norms (verbose only)
    if _verbose_this_step:
        print(f"\n--- step {step} grad diagnostics ---")
        _scalar("grad_norm_total", f"{grad_norm:.4g}")
        grad_norms = {}
        for path, g in tree_flatten(accum_grads):
            gnorm = float(mx.sum(g.astype(mx.float32) * g.astype(mx.float32)).item()) ** 0.5
            has_nan = bool(mx.any(mx.isnan(g)).item())
            flag = " *** NaN ***" if has_nan else ""
            if has_nan or gnorm > 0.1 * grad_norm or "A_log" in path or "dt_bias" in path or "out_proj" in path:
                print(f"  [grad/{path}] norm={gnorm:.6e}{flag}")
            grad_norms[path] = gnorm if not has_nan else "nan"
        _log_event("grads", {"grad_norm": grad_norm, "per_param": grad_norms})

    # Skip update if loss or gradients are NaN
    train_loss_f = float(train_loss.item())
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    _prev_loss_was_nan = math.isnan(grad_norm) or math.isnan(train_loss_f)
    if _prev_loss_was_nan:
        if VERBOSE:
            print(f"  *** SKIPPING UPDATE: loss={train_loss_f}, grad_norm={grad_norm} ***")
        mx.eval(model.parameters())  # still need to eval to keep graph in sync
    else:
        if grad_norm > max_grad_norm:
            clip_scale = max_grad_norm / grad_norm
            accum_grads = tree_map(lambda g: g * clip_scale, accum_grads)

        optimizer.set_lr_multiplier(lrm)
        optimizer.update(model, accum_grads)
        mx.eval(model.parameters(), *optimizer.state)

    if train_loss_f > 100:
        print("FAIL")
        raise SystemExit(1)

    dt = time.time() - t0
    if step >= STARTUP_EXCLUDE_STEPS:
        total_training_time += dt

    ema_beta = 0.9
    if not math.isnan(train_loss_f):
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    remaining = max(0.0, TIME_BUDGET - total_training_time)

    _log_event("step", {
        "loss_raw": train_loss_f, "loss_smooth": debiased_smooth_loss,
        "grad_norm": grad_norm, "lrm": lrm, "dt_ms": dt * 1000,
        "tok_per_sec": tok_per_sec, "skipped": math.isnan(grad_norm) or math.isnan(train_loss_f),
    })

    if USE_WANDB:
        wandb.log({
            "loss": train_loss_f, "loss_smooth": debiased_smooth_loss,
            "grad_norm": grad_norm, "lr_multiplier": lrm,
            "step_ms": dt * 1000, "tok_per_sec": tok_per_sec,
            "epoch": epoch,
        }, step=step)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step >= STARTUP_EXCLUDE_STEPS and total_training_time >= TIME_BUDGET:
        break

print()
t_train = time.time()
print(f"Training completed in {t_train - t_compiled:.1f}s")

total_tokens = step * TOTAL_BATCH_SIZE
print("Starting final eval...")
print(f"Final eval batch size: {FINAL_EVAL_BATCH_SIZE}")
val_bpb = evaluate_bpb(model, tokenizer, FINAL_EVAL_BATCH_SIZE)
t_eval = time.time()
print(f"Final eval completed in {t_eval - t_train:.1f}s")

steady_state_mfu = 0.0
peak_vram_mb = get_peak_memory_mb()

_log_event("result", {
    "val_bpb": val_bpb, "training_seconds": total_training_time,
    "total_seconds": t_eval - t_start, "peak_vram_mb": peak_vram_mb,
    "total_tokens_M": total_tokens / 1e6, "num_steps": step,
})
if _log_file is not None:
    _log_file.close()
    print(f"Verbose log saved to {_log_path}")

if USE_WANDB:
    wandb.log({"val_bpb": val_bpb, "peak_vram_mb": peak_vram_mb,
               "total_tokens_M": total_tokens / 1e6, "training_seconds": total_training_time})
    wandb.finish()

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
