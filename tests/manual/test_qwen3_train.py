"""
Qwen3 Training Test - Unified script supporting multiple configurations.

Supports:
  - Device: cuda (baseline) or flagos (optional FlagGems Triton kernels; disable with --no-flaggems)
  - Parallel: none (single GPU), ddp, or fsdp
  - Communication backend: nccl or flagcx (for ddp/fsdp)
  - Optional: device synchronize before each attention block forward (--qwen3-attention-device-presync)
  - On ``flagos``: optional swap of decoder ``Qwen3Attention`` for ``FlagosQwen3Attention`` (``--flagos-qwen3-attention``).

Usage:
    # Single GPU, pure CUDA baseline
    python tests/test_qwen3_train.py --device cuda

    # Single GPU, flagos (FlagGems)
    python tests/test_qwen3_train.py --device flagos

    # DDP with NCCL
    torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel ddp --comm nccl

    # DDP with FlagCX
    torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel ddp --comm flagcx

    # FSDP with NCCL
    torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel fsdp --comm nccl

    # FSDP with FlagCX
    torchrun --nproc_per_node=2 tests/test_qwen3_train.py --parallel fsdp --comm flagcx

    # NaN diagnostics (rank 0 only; samples a few tensors per step to limit overhead)
    python tests/test_qwen3_train.py --device flagos --debug-nan
    DEBUG_NAN=1 python tests/test_qwen3_train.py --device flagos

    # Backward: per-module grad g_in/g_out stats (mean,min,max,bad); first non-finite -> .pt in --debug-dump-dir
    python tests/test_qwen3_train.py --device flagos --debug-bwd-hooks --debug-dump-dir ./debug_dump
    # Quieter backward: summary only: add --debug-bwd-hooks-skip-per-hook-stats
    DEBUG_BWD_HOOKS=1 python tests/test_qwen3_train.py --device flagos --steps 1

    # self_attn I/O table per step; with --attention-probe or --debug-bwd-hooks, first forward NaN -> same dump dir
    python tests/test_qwen3_train.py --device flagos --attention-probe --debug-dump-dir ./debug_dump

    # flagos without FlagGems (PyTorch PrivateUse1 fallback; for debugging vs Triton kernels)
    python tests/test_qwen3_train.py --device flagos --no-flaggems
    TORCH_FLAGOS_DISABLE_FLAGGEMS=1 python tests/test_qwen3_train.py --device flagos

    # Before each Qwen3Attention forward: device sync (flagos.flagos.synchronize or torch.cuda.synchronize)
    python tests/manual/test_qwen3_train.py --device flagos --qwen3-attention-device-presync
    QWEN3_ATTENTION_DEVICE_PRESYNC=1 python tests/manual/test_qwen3_train.py --device flagos

    # Autograd engine smoke test on the training device (before loading the model; all ranks if distributed)
    python tests/manual/test_qwen3_train.py --device flagos --check-autograd-engine
    CHECK_AUTOGRAD_ENGINE=1 python tests/manual/test_qwen3_train.py --device flagos --steps 1

    # Qwen3Attention Q branch: q_proj / q_norm (+ k_norm) forward stats; optional backward absmax per step
    python tests/manual/test_qwen3_train.py --device flagos --debug-attn-q-path
    python tests/manual/test_qwen3_train.py --device flagos --debug-attn-q-path --debug-attn-q-path-layers 25,26,27
    DEBUG_ATTN_Q_PATH=1 DEBUG_ATTN_Q_PATH_BWD=1 python tests/manual/test_qwen3_train.py --device flagos

    # flagos: HF attention vs custom FlagosQwen3Attention (eager core as autograd.Function)
    python tests/manual/test_qwen3_train.py --device flagos --flagos-qwen3-attention custom   # default
    python tests/manual/test_qwen3_train.py --device flagos --flagos-qwen3-attention hf
    FLAGOS_QWEN3_ATTENTION=hf python tests/manual/test_qwen3_train.py --device flagos
"""

import argparse
import functools
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from dummy_dataset import DummyTextDataset

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 Training Test")
    parser.add_argument(
        "--device",
        choices=["cuda", "flagos"],
        default="flagos",
        help="Device type (default: flagos)",
    )
    parser.add_argument(
        "--parallel",
        choices=["none", "ddp", "fsdp"],
        default="none",
        help="Parallel strategy (default: none)",
    )
    parser.add_argument(
        "--comm",
        choices=["nccl", "flagcx"],
        default="nccl",
        help="Communication backend for distributed (default: nccl)",
    )
    parser.add_argument(
        "--model", default="/nfs/hcr/models/Qwen/Qwen3-0.6B", help="Model path"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length (default: 1024 for single, 256 for distributed)",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--debug-nan",
        action="store_true",
        help="Log finiteness stats for loss/logits, batch, sampled params/grads "
        "(rank 0). Also enabled if env DEBUG_NAN is 1/true/yes.",
    )
    parser.add_argument(
        "--debug-bwd-hooks",
        action="store_true",
        help="Step 1 only: full_backward_hook on every submodule; each hook prints "
        "grad_input/grad_output tensor stats (mean,min,max,bad); first non-finite "
        "backward saves tensors to --debug-dump-dir; enables first-forward-NaN capture "
        "for all steps (rank 0). Env DEBUG_BWD_HOOKS=1.",
    )
    parser.add_argument(
        "--debug-bwd-hooks-verbose",
        action="store_true",
        help="Print every backward-hook invocation (huge log). Default: anomalies + summary.",
    )
    parser.add_argument(
        "--debug-bwd-hooks-max-print",
        type=int,
        default=80,
        help="Max anomaly lines to print for --debug-bwd-hooks (default: 80).",
    )
    parser.add_argument(
        "--attention-probe",
        action="store_true",
        help="Rank 0: log self_attn I/O stats per block; also register full-model forward hooks "
        "to torch.save the first submodule whose output has non-finite values each step "
        "(see --debug-dump-dir). Env ATTENTION_PROBE=1.",
    )
    parser.add_argument(
        "--attention-probe-dir",
        type=str,
        default="attention_probe_dump",
        help="Legacy alias: used as --debug-dump-dir when the latter is unset.",
    )
    parser.add_argument(
        "--debug-dump-dir",
        type=str,
        default=None,
        help="Directory for first-NaN tensor saves (default: --attention-probe-dir value).",
    )
    parser.add_argument(
        "--debug-bwd-hooks-skip-per-hook-stats",
        action="store_true",
        help="With --debug-bwd-hooks, do not print mean/min/max/bad for every module (only summary).",
    )
    parser.add_argument(
        "--no-flaggems",
        action="store_true",
        help="For --device flagos: do not register FlagGems (PyTorch fallback for PrivateUse1; often slower). "
        "Same as env TORCH_FLAGOS_DISABLE_FLAGGEMS=1; must be in effect before importing torch_flagos "
        "(this script sets the env when you pass the flag).",
    )
    parser.add_argument(
        "--flagos-qwen3-attention",
        choices=("hf", "custom"),
        default="custom",
        help="For --device flagos only: keep HuggingFace Qwen3Attention (hf) or use FlagosQwen3Attention with "
        "custom eager autograd.Function (custom, default). Ignored when --device cuda. "
        "If env FLAGOS_QWEN3_ATTENTION is set to hf|huggingface|original or custom|flagos, it overrides this flag.",
    )
    parser.add_argument(
        "--qwen3-attention-device-presync",
        action="store_true",
        help="Register a forward_pre_hook on each Qwen3Attention that calls device synchronize before the module "
        "runs (torch_flagos.flagos.synchronize if --device flagos, else torch.cuda.synchronize). "
        "Env: QWEN3_ATTENTION_DEVICE_PRESYNC=1.",
    )
    parser.add_argument(
        "--check-autograd-engine",
        action="store_true",
        help="Before loading the model: run small autograd graphs on this process's training device "
        "(finite grads, expected numeric values; matmul + torch.autograd.grad). Catches many engine/stream "
        "read-order issues when paired with sync. Env CHECK_AUTOGRAD_ENGINE=1.",
    )
    parser.add_argument(
        "--debug-attn-q-path",
        action="store_true",
        help="Rank 0: after each forward, print per-layer stats for Qwen3Attention q_proj / q_norm outputs "
        "(and k_norm for contrast). Interprets forward vs backward when combined with --debug-attn-q-path-bwd. "
        "Env DEBUG_ATTN_Q_PATH=1.",
    )
    parser.add_argument(
        "--debug-attn-q-path-layers",
        type=str,
        default=None,
        help="Comma-separated decoder layer indices (e.g. 25,26,27). Default: all layers.",
    )
    parser.add_argument(
        "--debug-attn-q-path-bwd",
        action="store_true",
        help="With --debug-attn-q-path: register full_backward_hook on q_proj, q_norm, k_norm; "
        "after loss.backward(), print g_out/g_in absmax and non-finite flags (compact). "
        "Env DEBUG_ATTN_Q_PATH_BWD=1.",
    )
    args = parser.parse_args()
    env_debug = os.environ.get("DEBUG_NAN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    args.debug_nan = bool(args.debug_nan or env_debug)
    env_bwd = os.environ.get("DEBUG_BWD_HOOKS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    args.debug_bwd_hooks = bool(args.debug_bwd_hooks or env_bwd)
    env_attn = os.environ.get("ATTENTION_PROBE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    args.attention_probe = bool(args.attention_probe or env_attn)
    args.debug_dump_dir = args.debug_dump_dir or args.attention_probe_dir
    env_no_flaggems = os.environ.get("TORCH_FLAGOS_DISABLE_FLAGGEMS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    args.no_flaggems = bool(args.no_flaggems or env_no_flaggems)
    if args.no_flaggems:
        os.environ["TORCH_FLAGOS_DISABLE_FLAGGEMS"] = "1"

    _flagos_attn_env = os.environ.get("FLAGOS_QWEN3_ATTENTION", "").strip().lower()
    if _flagos_attn_env in ("hf", "huggingface", "original"):
        args.flagos_qwen3_attention = "hf"
    elif _flagos_attn_env in ("custom", "flagos"):
        args.flagos_qwen3_attention = "custom"

    def _env_truthy(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    args.qwen3_attention_device_presync = bool(
        args.qwen3_attention_device_presync or _env_truthy("QWEN3_ATTENTION_DEVICE_PRESYNC")
    )
    args.check_autograd_engine = bool(
        args.check_autograd_engine or _env_truthy("CHECK_AUTOGRAD_ENGINE")
    )
    args.debug_attn_q_path = bool(
        args.debug_attn_q_path or _env_truthy("DEBUG_ATTN_Q_PATH")
    )
    args.debug_attn_q_path_bwd = bool(
        args.debug_attn_q_path_bwd or _env_truthy("DEBUG_ATTN_Q_PATH_BWD")
    )

    if args.seq_len is None:
        args.seq_len = 256 if args.parallel != "none" else 1024
    return args


# ---------------------------------------------------------------------------
# Flagos: custom Qwen3 eager attention (autograd.Function fwd/bwd on device tensors)
# ---------------------------------------------------------------------------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA repeat (same as ``transformers.models.qwen3.modeling_qwen3.repeat_kv``)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class FlagosQwen3EagerAttentionFn(torch.autograd.Function):
    """Eager scaled dot-product attention (GQA + causal/additive mask + dropout) as one autograd node.

    Forward runs on the input device (e.g. flagos). Backward copies saved tensors and ``grad_output``
    to CPU for the matmul/softmax derivative chain, then moves ``grad_query``/``grad_key``/``grad_value``
    back to the forward device and dtype.
    """

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout_p: float,
        train_mode: bool,
        num_kv_groups: int,
    ) -> torch.Tensor:
        key_states = repeat_kv(key, num_kv_groups)
        value_states = repeat_kv(value, num_kv_groups)

        attn_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask

        attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query.dtype)

        dropout_mask: torch.Tensor | None
        if train_mode and dropout_p > 0.0:
            dropout_mask = torch.empty_like(attn_probs).bernoulli_(1.0 - dropout_p).div_(
                1.0 - dropout_p
            )
            attn_probs_used = attn_probs * dropout_mask
        else:
            dropout_mask = None
            attn_probs_used = attn_probs

        attn_output = torch.matmul(attn_probs_used, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        ctx.save_for_backward(query, key, value, attn_probs, attn_probs_used)
        ctx.scaling = scaling
        ctx.num_kv_groups = num_kv_groups
        ctx.dropout_mask = dropout_mask

        return attn_output

    @staticmethod
    def backward(ctx: Any, grad_attn_output: torch.Tensor):  # type: ignore[no-untyped-def]
        query, key, value, attn_probs, attn_probs_used = ctx.saved_tensors
        scaling = ctx.scaling
        num_kv_groups = ctx.num_kv_groups
        dropout_mask = ctx.dropout_mask

        out_dev = query.device
        out_dtype = query.dtype

        # Run backward math on CPU, then move grads back to the forward device (e.g. flagos).
        q = query.detach().cpu()
        k = key.detach().cpu()
        v = value.detach().cpu()
        ap = attn_probs.detach().cpu()
        apu = attn_probs_used.detach().cpu()
        go = grad_attn_output.contiguous().cpu()
        dm = dropout_mask.detach().cpu() if dropout_mask is not None else None

        key_states = repeat_kv(k, num_kv_groups)
        value_states = repeat_kv(v, num_kv_groups)

        grad_out_mat = go.transpose(1, 2).contiguous()

        grad_value_states = torch.matmul(apu.transpose(2, 3), grad_out_mat)
        grad_attn_probs_used = torch.matmul(grad_out_mat, value_states.transpose(2, 3))

        if dm is not None:
            grad_attn_probs = grad_attn_probs_used * dm
        else:
            grad_attn_probs = grad_attn_probs_used

        sum_grad = (ap * grad_attn_probs).sum(dim=-1, keepdim=True)
        grad_logits = ap * (grad_attn_probs - sum_grad)

        grad_query = torch.matmul(grad_logits, key_states) * scaling
        grad_key_states = torch.matmul(grad_logits.transpose(2, 3), q) * scaling

        b, h_kv, slen, d = k.shape
        if num_kv_groups == 1:
            grad_key = grad_key_states
            grad_value = grad_value_states
        else:
            grad_key = grad_key_states.view(b, h_kv, num_kv_groups, slen, d).sum(dim=2)
            grad_value = grad_value_states.view(b, h_kv, num_kv_groups, slen, d).sum(dim=2)

        grad_query = grad_query.to(device=out_dev, dtype=out_dtype)
        grad_key = grad_key.to(device=out_dev, dtype=out_dtype)
        grad_value = grad_value.to(device=out_dev, dtype=out_dtype)

        return grad_query, grad_key, grad_value, None, None, None, None, None


class FlagosQwen3Attention(torch.nn.Module):
    """Drop-in ``Qwen3Attention`` that routes the eager attention core through ``FlagosQwen3EagerAttentionFn``."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = torch.nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = torch.nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    @classmethod
    def from_hf(cls, src: Any) -> "FlagosQwen3Attention":
        m = cls(src.config, src.layer_idx)
        # New parameters default to CPU; load_state_dict would keep weights on CPU while
        # activations run on flagos/cuda. Move the empty module first, then load in-place.
        device = src.q_proj.weight.device
        m.to(device=device)
        m.load_state_dict(src.state_dict(), strict=True)
        return m

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        dropout_p = 0.0 if not self.training else self.attention_dropout
        attn_output = FlagosQwen3EagerAttentionFn.apply(
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.scaling,
            dropout_p,
            self.training,
            self.num_key_value_groups,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


def replace_self_attn_with_flagos_qwen3(model: torch.nn.Module) -> int:
    """Swap each HF ``Qwen3Attention`` under ``model.model.layers`` with ``FlagosQwen3Attention``."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    core = model.model if hasattr(model, "model") else model
    layers = getattr(core, "layers", None)
    if layers is None:
        raise AttributeError("Expected ``model.model.layers`` (CausalLM) or ``layers`` on core module.")
    n = 0
    for layer in layers:
        if isinstance(layer.self_attn, Qwen3Attention):
            layer.self_attn = FlagosQwen3Attention.from_hf(layer.self_attn)
            n += 1
        elif isinstance(layer.self_attn, FlagosQwen3Attention):
            pass
    return n


# ---------------------------------------------------------------------------
# Sync & printing utilities
# ---------------------------------------------------------------------------


def sync(args):
    if args.device == "flagos":
        import torch_flagos

        torch_flagos.flagos.synchronize()
    else:
        torch.cuda.synchronize()


def check_autograd_engine(args, device, rank, world_size):
    """Run minimal autograd cases on ``device`` to smoke-test the engine (every rank if distributed)."""
    tag = "[check-autograd]"

    def _fail(msg):
        raise RuntimeError(f"{tag} rank {rank}: {msg}")

    def _require_finite(t, name):
        if t is None:
            _fail(f"{name} is None")
        sync(args)
        if not torch.isfinite(t).all().item():
            bad = int((~torch.isfinite(t)).sum().item())
            _fail(f"{name} has {bad}/{t.numel()} non-finite values")

    # 1) backward() on a simple graph
    x = torch.randn(8, 8, device=device, dtype=torch.float32, requires_grad=True)
    y = (x * 3.0).sum()
    y.backward()
    _require_finite(x.grad, "case1(x*3).sum backward x.grad")
    if not torch.allclose(x.grad, torch.full_like(x, 3.0)):
        _fail("case1: x.grad should be all 3.0")

    # 2) torch.autograd.grad + matmul (different engine path; uses gemm backward)
    a = torch.randn(4, 5, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.randn(5, 6, device=device, dtype=torch.float32, requires_grad=True)
    ab = a @ b
    c = ab.sum()
    ga, gb = torch.autograd.grad(c, (a, b), create_graph=False)
    _require_finite(ga, "case2 matmul grad a")
    _require_finite(gb, "case2 matmul grad b")
    exp_ga = torch.ones_like(ab) @ b.t()
    exp_gb = a.t() @ torch.ones_like(ab)
    if not torch.allclose(ga, exp_ga):
        _fail("case2: grad w.r.t. a does not match ones @ b.T")
    if not torch.allclose(gb, exp_gb):
        _fail("case2: grad w.r.t. b does not match a.T @ ones")

    # 3) Two leaves, single backward
    u = torch.randn(3, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(3, device=device, dtype=torch.float32, requires_grad=True)
    w = (u + 2.0 * v).pow(2).sum()
    w.backward()
    _require_finite(u.grad, "case3 u.grad")
    _require_finite(v.grad, "case3 v.grad")
    exp_u = 2.0 * (u + 2.0 * v)
    exp_v = 4.0 * (u + 2.0 * v)
    if not torch.allclose(u.grad, exp_u):
        _fail("case3: u.grad mismatch (chain rule)")
    if not torch.allclose(v.grad, exp_v):
        _fail("case3: v.grad mismatch (chain rule)")

    # 4) retain_graph + second backward (light stress on queue bookkeeping)
    p = torch.randn(2, 2, device=device, dtype=torch.float32, requires_grad=True)
    q = p.sum()
    q.backward(retain_graph=True)
    _require_finite(p.grad, "case4 first backward p.grad")
    p.grad.zero_()
    (q * 2).backward()
    _require_finite(p.grad, "case4 second backward p.grad")
    if not torch.allclose(p.grad, torch.full_like(p, 2.0)):
        _fail("case4: second backward expected grad 2.0 everywhere")

    sync(args)
    if world_size > 1:
        dist.barrier()

    print_rank0(
        f"{tag} OK on device={device} (cases: backward, autograd.grad, two-leaf chain, retain_graph x2). "
        f"All ranks passed barrier."
        if world_size > 1
        else f"{tag} OK on device={device} (cases: backward, autograd.grad, two-leaf chain, retain_graph x2).",
        rank,
    )


def print_rank0(msg, rank):
    if rank == 0:
        print(msg)


# ---------------------------------------------------------------------------
# Optional NaN / finiteness diagnostics (--debug-nan, DEBUG_NAN=1)
# ---------------------------------------------------------------------------


def _debug_nan_param_name_substrings():
    """Substring hints to match a small subset of weights (keeps scans cheap)."""
    return (
        "embed_tokens",
        "layers.0",
        "layers.1",
        "lm_head",
        "model.norm",
    )


def _debug_nan_matches_param_name(name):
    return any(h in name for h in _debug_nan_param_name_substrings())


def debug_nan_tensor_stats(args, rank, tensor, label):
    """One-line summary: bad element count; if all finite, min/max/mean on device."""
    if not args.debug_nan or rank != 0:
        return
    if tensor is None:
        print_rank0(f"[debug-nan] {label}: None", rank)
        return
    sync(args)
    t = tensor.detach()
    bad = int((~torch.isfinite(t)).sum().item())
    n = t.numel()
    line = (
        f"[debug-nan] {label}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"bad={bad}/{n}"
    )
    if bad == 0:
        tf = t.float()
        line += (
            f" min={tf.min().item():.4e} max={tf.max().item():.4e} "
            f"mean={tf.mean().item():.4e}"
        )
    print_rank0(line, rank)


def debug_nan_loss(args, rank, step, when, loss):
    if not args.debug_nan or rank != 0:
        return
    sync(args)
    v = loss.detach()
    ok = bool(torch.isfinite(v).item())
    print_rank0(
        f"[debug-nan] step {step + 1} loss ({when}): finite={ok} value={v.item()}",
        rank,
    )


def debug_nan_batch(args, rank, step, batch):
    if not args.debug_nan or rank != 0:
        return
    for key in ("input_ids", "attention_mask", "labels"):
        if key not in batch:
            continue
        t = batch[key]
        if torch.is_tensor(t):
            debug_nan_tensor_stats(args, rank, t, f"step {step + 1} batch[{key}]")


def debug_nan_sample_params(args, rank, step, model, grads=False):
    """Check a few matching parameters (or their grads) for non-finite values."""
    if not args.debug_nan or rank != 0:
        return
    sync(args)
    label = "grad" if grads else "param"
    any_bad = False
    checked = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not _debug_nan_matches_param_name(name):
            continue
        checked += 1
        tensor = p.grad if grads else p.data
        if tensor is None:
            print_rank0(
                f"[debug-nan] step {step + 1} {label} {name}: tensor is None",
                rank,
            )
            continue
        bad = int((~torch.isfinite(tensor)).sum().item())
        if bad > 0:
            any_bad = True
            print_rank0(
                f"[debug-nan] step {step + 1} {label} {name}: bad={bad}/{tensor.numel()}",
                rank,
            )
    if checked == 0:
        print_rank0(
            f"[debug-nan] step {step + 1} {label}: no name matches for sample scan",
            rank,
        )
    elif not any_bad:
        print_rank0(
            f"[debug-nan] step {step + 1} {label}: sampled tensors OK ({checked} checked)",
            rank,
        )


# ---------------------------------------------------------------------------
# First-step backward tracing (--debug-bwd-hooks)
# ---------------------------------------------------------------------------


def _training_module_for_hooks(model):
    """DDP stores the real model in .module; FSDP/single-GPU use model as-is."""
    return model.module if hasattr(model, "module") else model


def install_qwen3_attention_forward_presync_hooks(model, args, rank):
    """If requested, sync device before each Qwen3 attention block forward (via ``register_forward_pre_hook``)."""
    if not getattr(args, "qwen3_attention_device_presync", False):
        return []
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    root = _training_module_for_hooks(model)
    handles: list = []

    def _pre_sync(_mod, _fwd_args, _fwd_kwargs):  # type: ignore[no-untyped-def]
        sync(args)

    for _name, mod in root.named_modules():
        if not isinstance(mod, (Qwen3Attention, FlagosQwen3Attention)):
            continue
        handles.append(mod.register_forward_pre_hook(_pre_sync, with_kwargs=True))

    print_rank0(
        f"[qwen3-attn-presync] forward_pre_hook(device sync) on {len(handles)} "
        "Qwen3 attention module(s) (HF or FlagosQwen3Attention)",
        rank,
    )
    return handles


def _grad_tuple_any_nonfinite(grad_tuple):
    if grad_tuple is None:
        return False
    for t in grad_tuple:
        if t is None or not torch.is_tensor(t):
            continue
        if not torch.isfinite(t).all().item():
            return True
    return False


def _finite_tensor_stats(t):
    """Scalar summary for one tensor (mean/min/max over finite elems; bad count)."""
    if t is None or not torch.is_tensor(t):
        return {}
    bad = int((~torch.isfinite(t)).sum().item())
    n = t.numel()
    tf = t.detach().float()
    if bad == n:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "bad": bad, "n": n}
    fin = torch.isfinite(tf)
    if not fin.any():
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "bad": bad, "n": n}
    v = tf[fin]
    return {
        "mean": float(v.mean().item()),
        "min": float(v.min().item()),
        "max": float(v.max().item()),
        "bad": bad,
        "n": n,
    }


def _format_grad_tuple_stats(tup, label):
    if tup is None:
        return f"{label}=(none)"
    parts = []
    for i, t in enumerate(tup):
        if t is None:
            parts.append(f"[{i}]=None")
        elif not torch.is_tensor(t):
            parts.append(f"[{i}]={type(t).__name__}")
        else:
            st = _finite_tensor_stats(t)
            parts.append(
                f"[{i}] mean={st['mean']:.4e} min={st['min']:.4e} max={st['max']:.4e} "
                f"bad={st['bad']}/{st['n']}"
            )
    return f"{label}: " + "; ".join(parts)


def _grad_tuple_finite_absmax(tup):
    """Max |x| over all finite elements in grad_input/grad_output tuple; -1 if none."""
    if tup is None:
        return -1.0
    m = -1.0
    for t in tup:
        if t is None or not torch.is_tensor(t):
            continue
        tf = t.detach().float()
        fin = torch.isfinite(tf)
        if not fin.any():
            continue
        m = max(m, float(tf[fin].abs().max().item()))
    return m


def _clone_grad_tuple_for_save(tup):
    if tup is None:
        return None
    out = []
    for t in tup:
        if t is None:
            out.append(None)
        elif torch.is_tensor(t):
            out.append(t.detach().cpu().contiguous())
        else:
            out.append(None)
    return tuple(out)


def debug_bwd_prepare_backward(state, step):
    if state is None:
        return
    state["bw_step"] = step
    state["bw_saved"] = False


def debug_bwd_hooks_register(model, args, rank):
    """Attach full backward hooks to all submodules; return (handles, state).

    Each hook prints grad_input / grad_output stats (mean, min, max, bad). The first
    hook that sees any non-finite grad saves tensors under ``args.debug_dump_dir``.
    """
    if not args.debug_bwd_hooks or rank != 0:
        return [], None

    target = _training_module_for_hooks(model)
    tag = f"[debug-bwd:{args.device}]"
    state = {
        "rows": [],
        "seq": 0,
        "bw_step": 0,
        "bw_saved": False,
        "tag": tag,
        "_ext_gout_abs": -1.0,
        "_ext_gin_abs": -1.0,
        "_ext_gout_seq": None,
        "_ext_gin_seq": None,
    }
    dump_dir = os.path.abspath(args.debug_dump_dir)
    os.makedirs(dump_dir, exist_ok=True)

    def make_hook(display_name, cls_name):
        def hook(module, grad_input, grad_output, *, _name=display_name, _cls=cls_name):
            state["seq"] += 1
            seq = state["seq"]
            go_bad = _grad_tuple_any_nonfinite(grad_output)
            gi_bad = _grad_tuple_any_nonfinite(grad_input)
            state["rows"].append((seq, _name, _cls, go_bad, gi_bad))

            gao = _grad_tuple_finite_absmax(grad_output)
            gai = _grad_tuple_finite_absmax(grad_input)
            if gao >= 0.0 and gao > state["_ext_gout_abs"]:
                state["_ext_gout_abs"] = gao
                state["_ext_gout_seq"] = (seq, _name, _cls)
            if gai >= 0.0 and gai > state["_ext_gin_abs"]:
                state["_ext_gin_abs"] = gai
                state["_ext_gin_seq"] = (seq, _name, _cls)

            t = state["tag"]
            if not args.debug_bwd_hooks_skip_per_hook_stats:
                gos = _format_grad_tuple_stats(grad_output, "g_out")
                gis = _format_grad_tuple_stats(grad_input, "g_in")
                print(f"{t} #{seq} {_cls} `{_name}` | {gos} | {gis}", flush=True)
            elif args.debug_bwd_hooks_verbose:
                print(
                    f"{t} #{seq} {_cls} `{_name}` "
                    f"grad_out_nonfinite={go_bad} grad_in_nonfinite={gi_bad}",
                    flush=True,
                )

            if (go_bad or gi_bad) and not state["bw_saved"]:
                state["bw_saved"] = True
                sync(args)
                step_human = int(state.get("bw_step", -1)) + 1
                safe_name = _name.replace("/", "_").replace(".", "_")[:200]
                path = os.path.join(
                    dump_dir,
                    f"first_backward_nonfinite_step{step_human}_seq{seq}_{safe_name}.pt",
                )
                torch.save(
                    {
                        "kind": "backward_first_nonfinite",
                        "train_step_0based": state.get("bw_step", -1),
                        "hook_seq": seq,
                        "module_name": _name,
                        "module_class": _cls,
                        "grad_output": _clone_grad_tuple_for_save(grad_output),
                        "grad_input": _clone_grad_tuple_for_save(grad_input),
                    },
                    path,
                )
                print_rank0(f"{t} saved first backward non-finite: {path}", rank)

        return hook

    handles = []
    for name, mod in target.named_modules():
        display = "(root)" if name == "" else name
        cls_name = type(mod).__name__
        try:
            h = mod.register_full_backward_hook(
                make_hook(display, cls_name), always_call=True
            )
        except TypeError:
            h = mod.register_full_backward_hook(make_hook(display, cls_name))
        handles.append(h)

    print_rank0(
        f"{tag} Registered {len(handles)} full_backward_hooks "
        f"(always_call=True) on `{type(target).__name__}`; "
        "per-hook grad stats print unless --debug-bwd-hooks-skip-per-hook-stats.",
        rank,
    )
    return handles, state


def debug_bwd_hooks_report(state, args, rank):
    if not args.debug_bwd_hooks or rank != 0 or state is None:
        return
    tag = state["tag"]
    rows = state["rows"]
    print_rank0(f"{tag} Total hook invocations: {len(rows)}", rank)

    if state["_ext_gout_abs"] >= 0.0:
        seq, name, cls = state["_ext_gout_seq"]
        print_rank0(
            f"{tag} Global max finite |grad_output| across hooks: "
            f"{state['_ext_gout_abs']:.4e} @ #{seq} {cls} `{name}`",
            rank,
        )
    if state["_ext_gin_abs"] >= 0.0:
        seq, name, cls = state["_ext_gin_seq"]
        print_rank0(
            f"{tag} Global max finite |grad_input| across hooks: "
            f"{state['_ext_gin_abs']:.4e} @ #{seq} {cls} `{name}`",
            rank,
        )

    first_go = next((r for r in rows if r[3]), None)
    first_gi_only = next((r for r in rows if r[4] and not r[3]), None)
    first_gi_any = next((r for r in rows if r[4]), None)

    if first_go:
        seq, name, cls, _, _ = first_go
        print_rank0(
            f"{tag} First grad_output non-finite: #{seq} {cls} `{name}` "
            "(NaN/Inf already present on gradient w.r.t. this module outputs — "
            "often produced in a later / loss-adjacent subgraph).",
            rank,
        )
    else:
        print_rank0(
            f"{tag} No hook saw non-finite grad_output "
            "(unusual if loss.backward had NaN).",
            rank,
        )

    if first_gi_only:
        seq, name, cls, _, _ = first_gi_only
        print_rank0(
            f"{tag} First grad_input non-finite with finite grad_output: "
            f"#{seq} {cls} `{name}` "
            "(strong hint: this module's backward or its direct autograd ops "
            "introduced non-finite grads).",
            rank,
        )
    elif first_gi_any and first_go:
        seq, name, cls, _, _ = first_gi_any
        print_rank0(
            f"{tag} First grad_input non-finite (grad_output already bad at #{first_go[0]}): "
            f"#{seq} {cls} `{name}`",
            rank,
        )

    anomalies = [r for r in rows if r[3] or r[4]]
    cap = max(0, args.debug_bwd_hooks_max_print)
    if (
        anomalies
        and not args.debug_bwd_hooks_verbose
        and args.debug_bwd_hooks_skip_per_hook_stats
    ):
        print_rank0(
            f"{tag} Anomaly lines (up to {cap} of {len(anomalies)}):", rank
        )
        for r in anomalies[:cap]:
            seq, name, cls, go_bad, gi_bad = r
            print_rank0(
                f"  #{seq} {cls} `{name}` "
                f"grad_out_nonfinite={go_bad} grad_in_nonfinite={gi_bad}",
                rank,
            )
        if len(anomalies) > cap:
            print_rank0(
                f"  ... {len(anomalies) - cap} more (raise --debug-bwd-hooks-max-print)",
                rank,
            )


def debug_bwd_hooks_remove(handles):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# First forward submodule with non-finite output (--attention-probe, rank 0)
# ---------------------------------------------------------------------------


def _structure_has_nonfinite(out):
    if out is None:
        return False
    if torch.is_tensor(out):
        return not torch.isfinite(out).all().item()
    if isinstance(out, (tuple, list)):
        return any(_structure_has_nonfinite(x) for x in out)
    return False


def _clone_for_save_structure(obj):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.detach().cpu().contiguous()
    if isinstance(obj, tuple):
        return tuple(_clone_for_save_structure(x) for x in obj)
    if isinstance(obj, list):
        return [_clone_for_save_structure(x) for x in obj]
    if isinstance(obj, (int, float, bool, str)):
        return obj
    return {"type": type(obj).__name__, "note": "skipped_non_tensor"}


def _clone_args_kwargs(args, kwargs):
    ca = tuple(_clone_for_save_structure(a) for a in (args or ()))
    ck = {k: _clone_for_save_structure(v) for k, v in (kwargs or {}).items()}
    return ca, ck


class FirstNanForwardCapture:
    """Full model tree: first submodule whose forward output contains non-finite -> save."""

    def __init__(self, args, rank):
        self.args = args
        self.rank = rank
        self.dump_dir = os.path.abspath(args.debug_dump_dir)
        self.current_step = 0
        self._pre = {}
        self._handles = []
        self._saved_this_step = False

    def before_forward(self, step):
        self.current_step = step
        self._saved_this_step = False
        self._pre.clear()

    @classmethod
    def install(cls, model, args, rank):
        if rank != 0:
            return None
        if not (args.attention_probe or args.debug_bwd_hooks):
            return None
        if args.parallel == "fsdp":
            print_rank0("[first-nan-fwd] skipped: FSDP not supported.", rank)
            return None

        inst = cls(args, rank)
        os.makedirs(inst.dump_dir, exist_ok=True)
        target = _training_module_for_hooks(model)
        n_mod = sum(1 for _ in target.named_modules())
        pre_kw_ok = True

        for name, mod in target.named_modules():
            disp = "(root)" if name == "" else name
            cls_name = type(mod).__name__

            def make_pre():
                def pre_hook(module, args, kwargs):
                    a = tuple(args) if args is not None else ()
                    kw = dict(kwargs) if kwargs else {}
                    inst._pre[id(module)] = (a, kw)

                return pre_hook

            def make_pre_legacy():
                def pre_hook(module, inp):
                    a = tuple(inp) if inp else ()
                    inst._pre[id(module)] = (a, {})

                return pre_hook

            def make_post(dn=disp, cn=cls_name):
                def post_hook(module, inp, out):
                    if inst._saved_this_step or not _structure_has_nonfinite(out):
                        return
                    inst._saved_this_step = True
                    sync(inst.args)
                    step_h = inst.current_step + 1
                    safe = dn.replace("/", "_").replace(".", "_")[:200]
                    path = os.path.join(
                        inst.dump_dir,
                        f"first_forward_nonfinite_step{step_h}_{safe}.pt",
                    )
                    pre_args, pre_kw = inst._pre.pop(id(module), ((), {}))
                    args_cpu, kw_cpu = _clone_args_kwargs(pre_args, pre_kw)
                    torch.save(
                        {
                            "kind": "forward_first_nonfinite",
                            "train_step_0based": inst.current_step,
                            "module_name": dn,
                            "module_class": cn,
                            "forward_args": args_cpu,
                            "forward_kwargs": kw_cpu,
                            "output": _clone_for_save_structure(out),
                        },
                        path,
                    )
                    print_rank0(f"[first-nan-fwd] saved {path}", inst.rank)

                return post_hook

            try:
                inst._handles.append(
                    mod.register_forward_pre_hook(make_pre(), with_kwargs=True)
                )
            except TypeError:
                if pre_kw_ok:
                    print_rank0(
                        "[first-nan-fwd] pre-hook without with_kwargs=True (limited capture).",
                        rank,
                    )
                    pre_kw_ok = False
                inst._handles.append(mod.register_forward_pre_hook(make_pre_legacy()))
            inst._handles.append(mod.register_forward_hook(make_post()))

        print_rank0(
            f"[first-nan-fwd] {n_mod} modules; first forward non-finite -> {inst.dump_dir}",
            rank,
        )
        return inst

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Per-block self_attn I/O stats only (--attention-probe)
# ---------------------------------------------------------------------------


class AttentionProbe:
    """Forward hooks on each `Qwen3DecoderLayer.self_attn` (rank 0): stats table only."""

    def __init__(self, args, rank):
        self.args = args
        self.rank = rank
        self.current_step = 0
        self._pre_inputs = {}
        self._rows = []
        self._handles = []

    @classmethod
    def install(cls, model, args, rank):
        if not args.attention_probe:
            return None
        if args.parallel == "fsdp":
            print_rank0(
                "[attention-probe] skipped: FSDP not supported (expects plain "
                "`model.model.layers`).",
                rank,
            )
            return None
        if rank != 0:
            return None

        inst = cls(args, rank)
        core = model.module if hasattr(model, "module") else model
        layers = core.model.layers
        n = len(layers)
        _pre_kw_ok = True
        for i, layer in enumerate(layers):
            try:
                inst._handles.append(
                    layer.self_attn.register_forward_pre_hook(
                        inst._make_pre_hook(i), with_kwargs=True
                    )
                )
            except TypeError:
                if _pre_kw_ok:
                    print_rank0(
                        "[attention-probe] pre-hook without with_kwargs=True "
                        "(upgrade PyTorch for keyword hidden_states capture).",
                        rank,
                    )
                    _pre_kw_ok = False
                inst._handles.append(
                    layer.self_attn.register_forward_pre_hook(
                        inst._make_pre_hook_legacy(i)
                    )
                )
            inst._handles.append(
                layer.self_attn.register_forward_hook(inst._make_post_hook(i))
            )
        print_rank0(f"[attention-probe] {n} self_attn blocks (forward I/O stats only)", rank)
        return inst

    def _make_pre_hook(self, layer_idx):
        def hook(module, args, kwargs):
            kw = kwargs or {}
            hs = kw.get("hidden_states")
            if hs is None and args:
                hs = args[0]
            self._pre_inputs[layer_idx] = hs

        return hook

    def _make_pre_hook_legacy(self, layer_idx):
        def hook(module, inp):
            hs = inp[0] if inp and len(inp) > 0 else None
            self._pre_inputs[layer_idx] = hs

        return hook

    def _make_post_hook(self, layer_idx):
        def hook(module, inp, out):
            hs = self._pre_inputs.pop(layer_idx, None)
            attn_out = out[0] if isinstance(out, tuple) else out
            si = _finite_tensor_stats(hs)
            so = _finite_tensor_stats(attn_out)
            self._rows.append((layer_idx, si, so))

        return hook

    def before_forward(self, step):
        self.current_step = step
        self._pre_inputs.clear()
        self._rows.clear()

    @staticmethod
    def _fmt_side(d):
        if not d:
            return "n/a"
        return (
            f"mean={d['mean']:.4e} min={d['min']:.4e} max={d['max']:.4e} "
            f"bad={d['bad']}/{d['n']}"
        )

    def flush(self):
        if self.rank != 0:
            return
        sync(self.args)
        self._rows.sort(key=lambda x: x[0])
        print_rank0(
            f"[attention-probe] step {self.current_step + 1} — self_attn I/O (all blocks, shallow→deep):",
            self.rank,
        )
        for layer_idx, si, so in self._rows:
            print_rank0(
                f"  L{layer_idx:02d}  in: {self._fmt_side(si)} | out: {self._fmt_side(so)}",
                self.rank,
            )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _parse_attn_q_path_layers(spec: str | None):
    """Return None (all layers) or a set of layer indices."""
    if spec is None or not str(spec).strip():
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


class AttnQPathProbe:
    """Forward (and optional backward) diagnostics on Qwen3Attention Q/K-norm branch.

    Forward: ``q_proj`` output, ``q_norm`` output, ``k_norm`` output (last for contrast).
    Backward: compact ``|grad_output|`` / ``|grad_input|`` abs-max for the same modules.
    """

    def __init__(self, args, rank, layer_filter: set[int] | None, with_bwd: bool):
        self.args = args
        self.rank = rank
        self.layer_filter = layer_filter
        self.with_bwd = with_bwd
        self.current_step = 0
        self._handles: list = []
        # layer_idx -> {"q_proj": stats, "q_norm": stats, "k_norm": stats}
        self._fwd: dict[int, dict[str, dict]] = {}
        self._bwd_rows: list[tuple] = []

    def _want_layer(self, layer_idx: int) -> bool:
        return self.layer_filter is None or layer_idx in self.layer_filter

    @classmethod
    def install(cls, model, args, rank):
        if not args.debug_attn_q_path or rank != 0:
            return None
        if args.parallel == "fsdp":
            print_rank0(
                "[attn-q-path] skipped: FSDP not supported (expects `model.model.layers`).",
                rank,
            )
            return None
        layer_filter = _parse_attn_q_path_layers(args.debug_attn_q_path_layers)
        inst = cls(args, rank, layer_filter, args.debug_attn_q_path_bwd)
        core = model.module if hasattr(model, "module") else model
        layers = core.model.layers
        n = len(layers)

        for i, layer in enumerate(layers):
            if not inst._want_layer(i):
                continue
            attn = layer.self_attn

            def qproj_fwd(mod, inp, out, *, li=i):
                inst._fwd.setdefault(li, {})["q_proj"] = _finite_tensor_stats(out)

            def qnorm_fwd(mod, inp, out, *, li=i):
                inst._fwd.setdefault(li, {})["q_norm"] = _finite_tensor_stats(out)

            def knorm_fwd(mod, inp, out, *, li=i):
                inst._fwd.setdefault(li, {})["k_norm"] = _finite_tensor_stats(out)

            inst._handles.append(attn.q_proj.register_forward_hook(qproj_fwd))
            inst._handles.append(attn.q_norm.register_forward_hook(qnorm_fwd))
            inst._handles.append(attn.k_norm.register_forward_hook(knorm_fwd))

            if inst.with_bwd:

                def bwd_qproj(mod, grad_input, grad_output, *, li=i):
                    inst._bwd_rows.append(
                        (
                            li,
                            "q_proj",
                            _grad_tuple_finite_absmax(grad_output),
                            _grad_tuple_finite_absmax(grad_input),
                            _grad_tuple_any_nonfinite(grad_output),
                            _grad_tuple_any_nonfinite(grad_input),
                        )
                    )

                def bwd_qnorm(mod, grad_input, grad_output, *, li=i):
                    inst._bwd_rows.append(
                        (
                            li,
                            "q_norm",
                            _grad_tuple_finite_absmax(grad_output),
                            _grad_tuple_finite_absmax(grad_input),
                            _grad_tuple_any_nonfinite(grad_output),
                            _grad_tuple_any_nonfinite(grad_input),
                        )
                    )

                def bwd_knorm(mod, grad_input, grad_output, *, li=i):
                    inst._bwd_rows.append(
                        (
                            li,
                            "k_norm",
                            _grad_tuple_finite_absmax(grad_output),
                            _grad_tuple_finite_absmax(grad_input),
                            _grad_tuple_any_nonfinite(grad_output),
                            _grad_tuple_any_nonfinite(grad_input),
                        )
                    )

                inst._handles.append(attn.q_proj.register_full_backward_hook(bwd_qproj))
                inst._handles.append(attn.q_norm.register_full_backward_hook(bwd_qnorm))
                inst._handles.append(attn.k_norm.register_full_backward_hook(bwd_knorm))

        filt = "all" if layer_filter is None else sorted(layer_filter)
        print_rank0(
            f"[attn-q-path] hooks on {n} layers (filter={filt}); "
            f"forward=q_proj,q_norm,k_norm; backward={'on' if inst.with_bwd else 'off'}",
            rank,
        )
        return inst

    def before_forward(self, step):
        self.current_step = step
        self._fwd.clear()

    @staticmethod
    def _fmt_st(d):
        if not d:
            return "n/a"
        return (
            f"mean={d['mean']:.4e} min={d['min']:.4e} max={d['max']:.4e} "
            f"bad={d['bad']}/{d['n']}"
        )

    def flush_forward(self):
        if self.rank != 0:
            return
        sync(self.args)
        step_h = self.current_step + 1
        print_rank0(
            f"[attn-q-path:fwd] step {step_h} — q_proj / q_norm / k_norm outputs "
            f"(post-module; q_norm is per-head RMSNorm on projected Q):",
            self.rank,
        )
        for li in sorted(self._fwd):
            d = self._fwd[li]
            qp = d.get("q_proj")
            qn = d.get("q_norm")
            kn = d.get("k_norm")
            line = (
                f"  L{li:02d}  q_proj: {self._fmt_st(qp)} | "
                f"q_norm: {self._fmt_st(qn)} | k_norm: {self._fmt_st(kn)}"
            )
            print_rank0(line, self.rank)
            hint = self._forward_hint(qp, qn, kn)
            if hint:
                print_rank0(f"       hint: {hint}", self.rank)

    @staticmethod
    def _forward_hint(qp, qn, kn):
        """Short heuristic; not a proof."""
        if not qp or not qn:
            return ""
        if qp["bad"] > 0:
            return "q_proj forward already has non-finite values — fix Q linear / upstream first."
        if qn["bad"] > 0:
            return "q_norm forward has non-finite values — norm or its input (view of q_proj) is suspect."
        try:
            qpm = max(abs(qp["min"]), abs(qp["max"]))
            qnm = max(abs(qn["min"]), abs(qn["max"]))
        except (TypeError, ValueError):
            return ""
        if qpm > 0 and qnm / qpm > 50:
            return (
                f"|q_norm|_inf scale / |q_proj|_inf scale ≈ {qnm / qpm:.1f} — "
                "large amplification across q_norm (forward)."
            )
        if kn and kn["bad"] == 0:
            try:
                knm = max(abs(kn["min"]), abs(kn["max"]))
            except (TypeError, ValueError):
                knm = 0.0
            if knm > 0 and qnm > 100 * knm:
                return (
                    "q_norm magnitudes much larger than k_norm — Q branch-specific forward issue likely."
                )
        return ""

    def flush_backward(self, step):
        if self.rank != 0 or not self.with_bwd:
            return
        if not self._bwd_rows:
            return
        sync(self.args)
        step_h = step + 1
        print_rank0(
            f"[attn-q-path:bwd] step {step_h} — full_backward_hook abs-max (finite elems) "
            f"and non-finite flags:",
            self.rank,
        )
        for li, name, go, gi, go_bad, gi_bad in sorted(
            self._bwd_rows, key=lambda r: (r[0], r[1])
        ):
            print_rank0(
                f"  L{li:02d}  {name:6s}  |g_out|_max={go:.4e}  |g_in|_max={gi:.4e}  "
                f"nonfinite_g_out={go_bad}  nonfinite_g_in={gi_bad}",
                self.rank,
            )
        self._interpret_fwd_vs_bwd(step_h)
        self._bwd_rows.clear()

    def _interpret_fwd_vs_bwd(self, step_h: int):
        """Compare last forward snapshot ``self._fwd`` with q_norm/q_proj backward rows."""
        by_layer: dict[int, dict[str, tuple]] = {}
        for li, name, go, gi, go_bad, gi_bad in self._bwd_rows:
            by_layer.setdefault(li, {})[name] = (go, gi, go_bad, gi_bad)

        lines = []
        for li in sorted(by_layer):
            fd = self._fwd.get(li, {})
            qn_f = fd.get("q_norm")
            qp_f = fd.get("q_proj")
            bqn = by_layer[li].get("q_norm")
            bqp = by_layer[li].get("q_proj")
            if not bqn or not qn_f:
                continue
            go, gi, go_bad, gi_bad = bqn
            try:
                fmax = max(abs(qn_f["min"]), abs(qn_f["max"]))
            except (TypeError, ValueError):
                fmax = float("nan")
            if (
                qn_f["bad"] == 0
                and fmax < 1e4
                and go > 1e12
                and not go_bad
            ):
                lines.append(
                    f"L{li:02d}: q_norm forward |activ|_max≈{fmax:.4e} but "
                    f"|dL/d(q_norm out)|_max≈{go:.4e} — gradient explosion likely from "
                    f"attention/SDP math or later ops, not from huge forward Q-norm values alone."
                )
            elif qn_f["bad"] > 0:
                lines.append(
                    f"L{li:02d}: q_norm forward already non-finite — investigate q_proj→view→q_norm forward."
                )
            elif bqp and qp_f and qp_f["bad"] == 0 and bqp[0] > 1e12 and not bqp[3]:
                try:
                    pq = max(abs(qp_f["min"]), abs(qp_f["max"]))
                except (TypeError, ValueError):
                    pq = float("nan")
                if pq < 1e3:
                    lines.append(
                        f"L{li:02d}: q_proj forward |out|_max≈{pq:.4e} but "
                        f"|dL/d(q_proj out)|_max≈{bqp[0]:.4e} — large grad w.r.t. Q projection despite "
                        f"moderate forward Q linear output."
                    )

        if not lines:
            return
        print_rank0(
            f"[attn-q-path:interpret] step {step_h} (heuristic; compare with full debug-bwd hooks):",
            self.rank,
        )
        for ln in lines[:24]:
            print_rank0(f"  • {ln}", self.rank)
        if len(lines) > 24:
            print_rank0(f"  • ... {len(lines) - 24} more layers", self.rank)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Device & distributed setup
# ---------------------------------------------------------------------------


def setup(args):
    """Initialize device and (optionally) distributed environment.

    Returns (device_str, local_rank, world_size, rank).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    if args.device == "flagos":
        import torch_flagos

        # Ensure allocator/runtime are up before any large .to(flagos) (avoids rare native crashes).
        torch_flagos.flagos.init()
        torch_flagos.flagos.set_device(local_rank)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(local_rank)

    if args.parallel == "none":
        return f"{args.device}:{local_rank}", local_rank, 1, 0

    # --- Distributed init ---
    if args.device == "flagos":
        import torch_flagos.distributed as flagos_dist

        flagos_dist.init_process_group(backend=args.comm)
    else:
        if args.comm == "flagcx":
            import flagcx  # noqa: F401

            dist.init_process_group(backend="cpu:gloo,cuda:flagcx")
        else:
            dist.init_process_group(backend="nccl")

    if rank == 0:
        pg = dist.distributed_c10d._get_default_group()
        print(f"[DEBUG] Backend config: {dist.get_backend_config()}", flush=True)
        print(f"[DEBUG] Process group device types: {pg._device_types}", flush=True)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = f"{args.device}:{local_rank}"
    return device, local_rank, world_size, rank


def cleanup(args):
    if args.parallel != "none":
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _print_word_embedding_diagnostics(model, rank):
    """Log tie_word_embeddings config and whether input/output embed weights are shared."""
    if rank != 0:
        return
    cfg = getattr(model, "config", None)
    tie = getattr(cfg, "tie_word_embeddings", None) if cfg is not None else None
    print_rank0(f"    config.tie_word_embeddings: {tie}", rank)
    try:
        inp = model.get_input_embeddings()
        out = model.get_output_embeddings()
        w_in = inp.weight if inp is not None else None
        w_out = out.weight if out is not None else None
    except Exception as exc:
        print_rank0(f"    [embed] get_input/output_embeddings failed: {exc}", rank)
        return
    if w_in is None or w_out is None:
        print_rank0(
            f"    [embed] missing weight (w_in={w_in is not None}, w_out={w_out is not None})",
            rank,
        )
        return
    print_rank0(
        f"    embed_tokens weight shape: {tuple(w_in.shape)} | lm_head weight shape: {tuple(w_out.shape)}",
        rank,
    )
    print_rank0(
        f"    weights tied: same Parameter object={w_in is w_out}, "
        f"same_storage(data_ptr)={w_in.data_ptr() == w_out.data_ptr()}",
        rank,
    )


def load_model(args, device, rank):
    """Load model and tokenizer, detect & freeze unused params."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print_rank0("\n[1] Loading model and tokenizer...", rank)
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(dtype=torch.float32, device_map="cpu")
    load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.device == "flagos":
        import gc

        import torch_flagos

        gc.collect()
        torch_flagos.flagos.synchronize()
        # One small allocation to flush driver/runtime before moving the full model.
        _warm = torch.zeros(1, device=device, dtype=torch.float32)
        torch_flagos.flagos.synchronize()
        del _warm

    print_rank0(f"    Moving weights to {device}...", rank)
    model = model.to(device)
    model.train()

    if args.device == "flagos":
        if args.flagos_qwen3_attention == "custom":
            n_swapped = replace_self_attn_with_flagos_qwen3(model)
            print_rank0(
                f"    FlagosQwen3Attention: replaced {n_swapped} Qwen3Attention module(s) "
                "(eager matmul/softmax/dropout as custom autograd.Function).",
                rank,
            )
        else:
            print_rank0(
                "    Qwen3Attention: using HuggingFace implementation (--flagos-qwen3-attention hf).",
                rank,
            )

    if args.device == "flagos" and hasattr(model, "tie_weights"):
        if getattr(model.config, "tie_word_embeddings", False):
            model.tie_weights()
            print_rank0(
                "    model.tie_weights() after .to(flagos) "
                "(restore embed_tokens/lm_head sharing if broken by device move).",
                rank,
            )

    # Detect and freeze unused parameters
    print_rank0("\n[1.5] Detecting and freezing unused parameters...", rank)
    dummy_input = torch.randint(0, 1000, (1, 32), device=device)
    with torch.enable_grad():
        out = model(
            input_ids=dummy_input, attention_mask=None, labels=None, use_cache=False
        )
        out.logits.sum().backward()

    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            param.requires_grad = False
            unused_params.append(name)
        else:
            param.grad = None

    if args.debug_nan:
        debug_nan_tensor_stats(
            args, rank, out.logits, "[1.5] dummy forward logits (unused-param probe)"
        )

    print_rank0(f"    Frozen {len(unused_params)} unused parameters", rank)
    if rank == 0 and unused_params:
        for name in unused_params[:5]:
            print(f"      - {name}")
        if len(unused_params) > 5:
            print(f"      ... and {len(unused_params) - 5} more")

    sync(args)
    print_rank0(f"Model device: {next(model.parameters()).device}", rank)
    print_rank0(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
        rank,
    )
    print_rank0(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M",
        rank,
    )
    print_rank0("    --- word embedding / lm_head tie diagnostics ---", rank)
    _print_word_embedding_diagnostics(model, rank)
    print_rank0(f"Model load time: {time.time() - load_start:.2f}s", rank)

    return model, tokenizer


# ---------------------------------------------------------------------------
# DDP wrapping
# ---------------------------------------------------------------------------


def wrap_ddp(model, args, local_rank, rank):
    """Wrap model with DDP."""
    if args.device == "flagos":
        import torch_flagos.distributed as flagos_dist

        model = flagos_dist.DistributedDataParallel(model)
        print_rank0("    DDP: flagos mode (python_reducer + custom grad hooks)", rank)
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[local_rank])
        print_rank0("    DDP: standard mode (CUDA)", rank)
    return model


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------


def wrap_fsdp(model, args, device, rank):
    """Wrap model with FSDP."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.device(device),
        use_orig_params=True,
    )
    model.train()

    # Validate: detect unused parameters via dummy forward+backward
    print_rank0("\n[1.5b] Validating FSDP gradient flow...", rank)
    dummy_input = torch.randint(0, 1000, (1, 32), device=device)
    with torch.enable_grad():
        out = model(input_ids=dummy_input, use_cache=False)
        out.logits.sum().backward()

    unused = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    print_rank0(f"    Parameters without gradient: {len(unused)}", rank)
    model.zero_grad(set_to_none=True)

    print_rank0(f"    FSDP: FULL_SHARD (device={args.device})", rank)
    return model


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


def create_dataloader(args, tokenizer, world_size, rank):
    """Create dataloader (with DistributedSampler if distributed)."""
    dataset = DummyTextDataset(tokenizer, num_samples=100, max_length=args.seq_len)
    sampler = None
    if args.parallel != "none":
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )
    return dataloader, sampler


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    model,
    batch,
    device,
    args,
    step=0,
    attention_probe=None,
    first_nan_forward=None,
    attn_q_path_probe=None,
):
    """Forward + loss computation.

    Returns (loss, batch_tokens, forward_outputs_or_none).
    ``forward_outputs`` is only retained when ``args.debug_nan`` to inspect logits.
    """
    if attention_probe is not None:
        attention_probe.before_forward(step)
    if first_nan_forward is not None:
        first_nan_forward.before_forward(step)
    if attn_q_path_probe is not None:
        attn_q_path_probe.before_forward(step)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    loss = outputs.loss

    if attention_probe is not None:
        attention_probe.flush()
    if attn_q_path_probe is not None:
        attn_q_path_probe.flush_forward()

    fwd = outputs if args.debug_nan else None
    return loss, input_ids.numel(), fwd


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(args, step_times, total_loss, total_tokens, world_size, rank):
    label_parts = []
    if args.device == "flagos":
        import torch_flagos

        label_parts.append(
            "flagos + FlagGems"
            if torch_flagos.is_flaggems_enabled()
            else "flagos (no FlagGems)"
        )
    else:
        label_parts.append("Pure CUDA")
    if args.parallel != "none":
        label_parts.append(args.parallel.upper())
    if args.parallel != "none":
        label_parts.append(args.comm.upper())
    label = " | ".join(label_parts)

    print_rank0("\n" + "=" * 60, rank)
    print_rank0(f"Training Summary ({label}):", rank)
    if args.parallel != "none":
        print_rank0(f"  World size: {world_size} GPUs", rank)
    print_rank0(f"  Total training steps: {args.steps}", rank)
    print_rank0(f"  Average loss: {total_loss / args.steps:.4f}", rank)
    if args.parallel != "none":
        print_rank0(f"  Total tokens (per GPU): {total_tokens}", rank)
        print_rank0(f"  Total tokens (all GPUs): {total_tokens * world_size}", rank)
    else:
        print_rank0(f"  Total tokens: {total_tokens}", rank)
    print_rank0("-" * 60, rank)

    tokens_per_step = args.batch_size * args.seq_len
    suffix = " per GPU" if args.parallel != "none" else ""

    if len(step_times) > 1:
        first = step_times[0]
        rest = step_times[1:]
        avg = sum(rest) / len(rest)
        print_rank0(
            f"  First step: {first:.2f}s ({tokens_per_step / first:.1f} tokens/s{suffix})",
            rank,
        )
        print_rank0(
            f"  Average subsequent steps: {avg:.2f}s ({tokens_per_step / avg:.1f} tokens/s{suffix})",
            rank,
        )
    else:
        avg = step_times[0]
        print_rank0(
            f"  Average per step: {avg:.2f}s ({tokens_per_step / avg:.1f} tokens/s{suffix})",
            rank,
        )

    print_rank0("-" * 60, rank)
    total_time = sum(step_times)
    print_rank0(f"  Total training time: {total_time:.2f}s", rank)
    print_rank0(
        f"  Overall throughput{suffix}: {total_tokens / total_time:.1f} tokens/s", rank
    )
    if args.parallel != "none":
        print_rank0(
            f"  Overall throughput (all GPUs): {total_tokens * world_size / total_time:.1f} tokens/s",
            rank,
        )
    print_rank0("=" * 60, rank)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # --- Setup ---
    device, local_rank, world_size, rank = setup(args)

    label_parts = [args.device.upper()]
    if args.parallel != "none":
        label_parts.append(args.parallel.upper())
        label_parts.append(args.comm.upper())

    print_rank0("=" * 60, rank)
    print_rank0(f"Qwen3 Training Test [{' | '.join(label_parts)}]", rank)
    print_rank0("=" * 60, rank)

    if args.device == "flagos":
        import torch_flagos

        print_rank0(
            f"Flagos device available: {torch_flagos.flagos.is_available()}", rank
        )
        print_rank0(f"FlagGems registered: {torch_flagos.is_flaggems_enabled()}", rank)
        print_rank0(
            f"Registered ops count: {len(torch_flagos.get_registered_ops())}", rank
        )
        if args.no_flaggems or not torch_flagos.is_flaggems_enabled():
            print_rank0(
                "FlagGems off: using PyTorch PrivateUse1 fallback (not Triton). "
                "TORCH_FLAGOS_DISABLE_FLAGGEMS or --no-flaggems.",
                rank,
            )
        print_rank0(
            f"Qwen3 attention mode: {args.flagos_qwen3_attention} "
            "(hf=HuggingFace, custom=FlagosQwen3Attention; --flagos-qwen3-attention / FLAGOS_QWEN3_ATTENTION)",
            rank,
        )
    else:
        print_rank0(f"CUDA available: {torch.cuda.is_available()}", rank)
        if torch.cuda.is_available():
            print_rank0(f"CUDA device: {torch.cuda.get_device_name(local_rank)}", rank)

    if args.parallel != "none":
        print_rank0(
            f"World size: {world_size}, rank: {rank}, local_rank: {local_rank}", rank
        )

    if args.debug_nan:
        print_rank0(
            "[debug-nan] Enabled: extra finiteness logs on rank 0 "
            "(sampled params/grads; full model not scanned).",
            rank,
        )
    if args.debug_bwd_hooks:
        print_rank0(
            f"[debug-bwd] Enabled (device={args.device}): step 1 only, "
            "per-module backward hooks on rank 0 (g_in/g_out mean/min/max per hook).",
            rank,
        )
    if args.attention_probe:
        print_rank0(
            "[attention-probe] Enabled on rank 0: per-block self_attn stats + layer dump.",
            rank,
        )
    if args.qwen3_attention_device_presync:
        print_rank0(
            "[qwen3-attn-presync] Will install forward_pre_hook(sync) on each Qwen3 attention block after wrap.",
            rank,
        )
    if args.debug_attn_q_path:
        print_rank0(
            "[attn-q-path] Enabled on rank 0: q_proj/q_norm/k_norm forward; "
            "add --debug-attn-q-path-bwd for compact backward absmax lines.",
            rank,
        )
    if args.check_autograd_engine:
        print_rank0(
            "[check-autograd] Running engine smoke test on each rank's device (before model load)...",
            rank,
        )
        check_autograd_engine(args, device, rank, world_size)

    # --- Load model ---
    model, tokenizer = load_model(args, device, rank)

    # --- Distributed barrier before wrapping ---
    if args.parallel != "none":
        sync(args)
        t = torch.zeros(1, device=device)
        dist.all_reduce(t)
        sync(args)

    # --- Wrap model ---
    if args.parallel == "ddp":
        model = wrap_ddp(model, args, local_rank, rank)
    elif args.parallel == "fsdp":
        model = wrap_fsdp(model, args, device, rank)

    install_qwen3_attention_forward_presync_hooks(model, args, rank)

    attention_probe = AttentionProbe.install(model, args, rank)
    first_nan_forward = FirstNanForwardCapture.install(model, args, rank)
    attn_q_path_probe = AttnQPathProbe.install(model, args, rank)

    # --- DataLoader ---
    print_rank0("\n[2] Creating dataset...", rank)
    dataloader, sampler = create_dataloader(args, tokenizer, world_size, rank)
    print_rank0(f"Dataset size: {len(dataloader.dataset)}", rank)
    print_rank0(
        f"Batch size{' per GPU' if args.parallel != 'none' else ''}: {args.batch_size}",
        rank,
    )
    if args.parallel != "none":
        print_rank0(f"Global batch size: {args.batch_size * world_size}", rank)
    print_rank0(f"Sequence length: {args.seq_len}", rank)

    # --- Optimizer ---
    print_rank0("\n[3] Creating optimizer...", rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print_rank0(f"Optimizer: AdamW, lr={args.lr}", rank)

    # --- Training loop ---
    parallel_label = f" {args.parallel.upper()}" if args.parallel != "none" else ""
    print_rank0(
        f"\n[4] Starting{parallel_label} training ({args.steps} steps)...", rank
    )

    total_tokens = 0
    total_loss = 0.0
    step_times = []

    if sampler is not None:
        sampler.set_epoch(0)
    data_iter = iter(dataloader)

    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step + 1)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        sync(args)
        step_start = time.time()

        if args.debug_nan:
            debug_nan_batch(args, rank, step, batch)

        bwd_handles, bwd_state = (
            debug_bwd_hooks_register(model, args, rank)
            if (args.debug_bwd_hooks and step == 0)
            else ([], None)
        )

        loss, batch_tokens, fwd_out = train_step(
            model,
            batch,
            device,
            args,
            step=step,
            attention_probe=attention_probe,
            first_nan_forward=first_nan_forward,
            attn_q_path_probe=attn_q_path_probe,
        )

        if args.debug_nan:
            debug_nan_loss(args, rank, step, "after forward", loss)
            if fwd_out is not None:
                debug_nan_tensor_stats(
                    args, rank, fwd_out.logits, f"step {step + 1} logits"
                )

        debug_bwd_prepare_backward(bwd_state, step)
        loss.backward()

        if attn_q_path_probe is not None:
            attn_q_path_probe.flush_backward(step)

        if bwd_handles:
            sync(args)
            debug_bwd_hooks_report(bwd_state, args, rank)
            debug_bwd_hooks_remove(bwd_handles)

        if args.debug_nan:
            debug_nan_sample_params(args, rank, step, model, grads=True)

        optimizer.step()

        if args.debug_nan:
            debug_nan_sample_params(args, rank, step, model, grads=False)

        optimizer.zero_grad()

        sync(args)
        step_time = time.time() - step_start
        step_times.append(step_time)

        total_tokens += batch_tokens
        total_loss += loss.item()

        print_rank0(
            f"  Step {step + 1}/{args.steps}: "
            f"loss={loss.item():.4f}, time={step_time:.2f}s, "
            f"tokens/s={batch_tokens / step_time:.1f}",
            rank,
        )

    # --- Summary ---
    print_summary(args, step_times, total_loss, total_tokens, world_size, rank)
    print_rank0("\nTraining test completed!", rank)

    if attention_probe is not None:
        attention_probe.remove()
    if first_nan_forward is not None:
        first_nan_forward.remove()
    if attn_q_path_probe is not None:
        attn_q_path_probe.remove()

    cleanup(args)


if __name__ == "__main__":
    main()
