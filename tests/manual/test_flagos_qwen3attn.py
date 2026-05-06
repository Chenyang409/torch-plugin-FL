#!/usr/bin/env python3
"""
用 ``layer0_attn_fixture.pt`` 驱动 **Qwen3Attention**，在 **三个独立子进程** 中分别跑
**CPU / CUDA / flagos**，主进程汇总 ``attn_output`` 与可选 ``hidden_states.grad`` 的 min/max/mean。

子进程互不共享 Python 解释器状态，避免混用设备时 PyTorch autograd 流断言等问题。

用法::

    python tests/manual/test_flagos_qwen3attn.py --allow-flaggems
    python tests/manual/test_flagos_qwen3attn.py --allow-flaggems --no-bwd

内部子进程入口（勿手动调用）::

    python tests/manual/test_flagos_qwen3attn.py --worker-backend cpu
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO_ROOT / "layer0_attn_fixture.pt"
DEFAULT_MODEL = "/data/nfs/Qwen3-0.6B"


def _device_is_flagos(d: torch.device) -> bool:
    t = getattr(d, "type", "") or ""
    return t == "flagos" or str(d).lower().startswith("flagos")


def _tensor_min_max_mean(t: torch.Tensor) -> tuple[float, float, float]:
    x = t.detach().to(dtype=torch.float64).flatten()
    finite = torch.isfinite(x)
    if not finite.any():
        return float("nan"), float("nan"), float("nan")
    xf = x[finite]
    return xf.min().item(), xf.max().item(), xf.mean().item()


def _print_stats_table(title: str, rows: list[tuple[str, float, float, float]]) -> None:
    print(title)
    print(f"{'backend':<10} {'min':>14} {'max':>14} {'mean':>14}")
    for name, lo, hi, mu in rows:
        print(f"{name:<10} {lo:14.6e} {hi:14.6e} {mu:14.6e}")


def _load_fixture(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"fixture not found: {path}")
    bundle = torch.load(str(path), map_location="cpu", weights_only=False)
    need = ("hidden_states", "cos", "sin", "config", "attn_state_dict")
    missing = [k for k in need if k not in bundle]
    if missing:
        raise KeyError(f"fixture missing keys: {missing}")
    return bundle


def _config_from_bundle(bundle: dict[str, Any], model_fallback: str) -> Any:
    from transformers import AutoConfig

    try:
        return AutoConfig.from_dict(bundle["config"])
    except Exception:
        return AutoConfig.from_pretrained(model_fallback)


def _apply_attn_eager(config: Any) -> None:
    if hasattr(config, "_attn_implementation"):
        try:
            object.__setattr__(config, "_attn_implementation", "eager")
        except Exception:
            setattr(config, "_attn_implementation", "eager")
    elif hasattr(config, "attn_implementation"):
        setattr(config, "attn_implementation", "eager")


class CpuFallbackQwen3Attention(torch.nn.Module):
    """CPU：HF ``Qwen3Attention`` + fixture 权重，固定 cpu。"""

    def __init__(self, config: Any, state_dict: dict[str, torch.Tensor], layer_idx: int = 0) -> None:
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        self._attn = Qwen3Attention(config, layer_idx=layer_idx)
        self._attn.load_state_dict(state_dict)
        self._attn.to(device=torch.device("cpu"), dtype=torch.float32)
        self._attn.eval()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any,
    ) -> tuple[torch.Tensor, Any]:
        return self._attn(hidden_states, position_embeddings, attention_mask, past_key_values)


def _build_hf_attention(
    bundle: dict[str, Any],
    device: torch.device,
    config: Any,
) -> torch.nn.Module:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    attn = Qwen3Attention(config, layer_idx=0)
    attn.load_state_dict(bundle["attn_state_dict"])
    attn.to(device=device, dtype=torch.float32)
    attn.eval()
    return attn


def _load_inputs(
    bundle: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    hs = bundle["hidden_states"].to(device=device, dtype=torch.float32)
    cos = bundle["cos"].to(device=device, dtype=torch.float32)
    sin = bundle["sin"].to(device=device, dtype=torch.float32)
    am = bundle.get("attention_mask")
    if am is not None:
        am = am.to(device=device)
    return hs, cos, sin, am


def _forward_attn_output(
    attn: torch.nn.Module,
    hs: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    am: torch.Tensor | None,
) -> torch.Tensor:
    with torch.no_grad():
        out, _ = attn(hs, (cos, sin), am, None)
    return out


def _grad_out_template(bundle: dict[str, Any], ref_attn_cpu: torch.Tensor) -> torch.Tensor:
    go = bundle.get("grad_attn_output")
    if go is None:
        return torch.ones_like(ref_attn_cpu)
    go = go.to(dtype=torch.float32)
    if tuple(go.shape) != tuple(ref_attn_cpu.shape):
        raise ValueError(
            f"grad_attn_output shape {tuple(go.shape)} != attn_output {tuple(ref_attn_cpu.shape)}"
        )
    return go


def _hidden_states_grad(
    bundle: dict[str, Any],
    device: torch.device,
    config: Any,
    grad_out_cpu: torch.Tensor,
    torch_flagos_mod: Any | None,
) -> torch.Tensor:
    attn = _build_hf_attention(bundle, device, config)
    hs, cos, sin, am = _load_inputs(bundle, device)
    grad_out = grad_out_cpu.to(device=device, dtype=torch.float32)
    hs = hs.clone().detach().requires_grad_(True)
    cos, sin = cos.detach(), sin.detach()
    if am is not None:
        am = am.detach()
    out, _ = attn(hs, (cos, sin), am, None)
    (g_hs,) = torch.autograd.grad(out, hs, grad_outputs=grad_out, retain_graph=False)
    if _device_is_flagos(device) and torch_flagos_mod is not None:
        torch_flagos_mod.flagos.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    return g_hs.cpu().clone()


def _nan_triple() -> tuple[float, float, float]:
    return float("nan"), float("nan"), float("nan")


def run_worker_backend(args: argparse.Namespace) -> int:
    """单后端子进程：向 stdout 打印一行 JSON（stderr 可含警告）。"""
    backend = args.worker_backend
    assert backend in ("cpu", "cuda", "flagos")

    try:
        from sqlalchemy.exc import SAWarning as _SAWarning

        warnings.simplefilter("ignore", _SAWarning)
    except ImportError:
        pass

    bundle = _load_fixture(args.fixture)
    config = _config_from_bundle(bundle, args.model)
    if not args.no_attn_eager:
        _apply_attn_eager(config)

    torch_flagos_mod: Any | None = None
    device: torch.device

    if backend == "cpu":
        cpu_mod = CpuFallbackQwen3Attention(config, bundle["attn_state_dict"])
        hs, cos, sin, am = _load_inputs(bundle, torch.device("cpu"))
        out = _forward_attn_output(cpu_mod, hs, cos, sin, am).cpu()
    elif backend == "cuda":
        if not torch.cuda.is_available():
            print(
                json.dumps(
                    {
                        "backend": backend,
                        "ok": False,
                        "error": "cuda_not_available",
                        "fwd": list(_nan_triple()),
                        "bwd": None if args.no_bwd else list(_nan_triple()),
                    }
                ),
                flush=True,
            )
            return 1
        attn = _build_hf_attention(bundle, torch.device("cuda"), config)
        hs, cos, sin, am = _load_inputs(bundle, torch.device("cuda"))
        out = _forward_attn_output(attn, hs, cos, sin, am)
        torch.cuda.synchronize()
        out = out.cpu()
    else:
        if not args.allow_flaggems:
            os.environ["TORCH_FLAGOS_DISABLE_FLAGGEMS"] = "1"
        try:
            import torch_flagos
        except ImportError as exc:
            print(
                json.dumps(
                    {
                        "backend": backend,
                        "ok": False,
                        "error": f"import_torch_flagos:{exc}",
                        "fwd": list(_nan_triple()),
                        "bwd": None if args.no_bwd else list(_nan_triple()),
                    }
                ),
                flush=True,
            )
            return 1
        if not torch_flagos.flagos.is_available():
            print(
                json.dumps(
                    {
                        "backend": backend,
                        "ok": False,
                        "error": "flagos_not_available",
                        "fwd": list(_nan_triple()),
                        "bwd": None if args.no_bwd else list(_nan_triple()),
                    }
                ),
                flush=True,
            )
            return 1
        torch_flagos_mod = torch_flagos
        torch_flagos.flagos.set_device(0)
        device = torch.device("flagos:0")
        attn = _build_hf_attention(bundle, device, config)
        hs, cos, sin, am = _load_inputs(bundle, device)
        out = _forward_attn_output(attn, hs, cos, sin, am)
        torch_flagos.flagos.synchronize()
        out = out.cpu()

    fwd_stats = list(_tensor_min_max_mean(out))
    bwd_stats: list[float] | None = None
    if not args.no_bwd:
        grad_tpl = _grad_out_template(bundle, out)
        if backend == "cpu":
            g = _hidden_states_grad(bundle, torch.device("cpu"), config, grad_tpl, None)
        elif backend == "cuda":
            g = _hidden_states_grad(bundle, torch.device("cuda"), config, grad_tpl, None)
        else:
            g = _hidden_states_grad(bundle, torch.device("flagos:0"), config, grad_tpl, torch_flagos_mod)
        bwd_stats = list(_tensor_min_max_mean(g))

    print(
        json.dumps(
            {
                "backend": backend,
                "ok": True,
                "fwd": fwd_stats,
                "bwd": bwd_stats,
            }
        ),
        flush=True,
    )
    return 0


def _build_worker_cmd(script: Path, backend: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(script),
        "--worker-backend",
        backend,
        "--fixture",
        str(args.fixture),
        "--model",
        args.model,
    ]
    if args.allow_flaggems:
        cmd.append("--allow-flaggems")
    if args.no_attn_eager:
        cmd.append("--no-attn-eager")
    if args.no_bwd:
        cmd.append("--no-bwd")
    return cmd


def _run_subprocess_worker(script: Path, backend: str, args: argparse.Namespace) -> dict[str, Any]:
    cmd = _build_worker_cmd(script, backend, args)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    out_lines = [ln for ln in (proc.stdout or "").strip().splitlines() if ln.strip()]
    err = (proc.stderr or "").strip()
    if err:
        for ln in err.splitlines():
            print(ln, file=sys.stderr)
    payload: dict[str, Any]
    if not out_lines:
        payload = {
            "backend": backend,
            "ok": False,
            "error": f"no_stdout rc={proc.returncode}",
            "fwd": list(_nan_triple()),
            "bwd": None if args.no_bwd else list(_nan_triple()),
        }
        return payload
    try:
        payload = json.loads(out_lines[-1])
    except json.JSONDecodeError:
        payload = {
            "backend": backend,
            "ok": False,
            "error": "invalid_json",
            "fwd": list(_nan_triple()),
            "bwd": None if args.no_bwd else list(_nan_triple()),
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen3Attention: CPU / CUDA / flagos in isolated subprocesses, then aggregate stats"
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--allow-flaggems",
        action="store_true",
        help="Do not set TORCH_FLAGOS_DISABLE_FLAGGEMS=1 in flagos worker",
    )
    parser.add_argument("--no-attn-eager", action="store_true")
    parser.add_argument("--no-bwd", action="store_true")
    parser.add_argument(
        "--worker-backend",
        choices=("cpu", "cuda", "flagos"),
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.worker_backend is not None:
        return run_worker_backend(args)

    script = Path(__file__).resolve()
    backends = ("cpu", "cuda", "flagos")
    results: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_run_subprocess_worker, script, b, args): b for b in backends}
        for fut in as_completed(futs):
            b = futs[fut]
            try:
                results[b] = fut.result()
            except Exception as exc:
                results[b] = {
                    "backend": b,
                    "ok": False,
                    "error": str(exc),
                    "fwd": list(_nan_triple()),
                    "bwd": None if args.no_bwd else list(_nan_triple()),
                }

    def row(backend: str) -> tuple[str, float, float, float]:
        p = results.get(backend, {})
        t = p.get("fwd") or list(_nan_triple())
        return backend, float(t[0]), float(t[1]), float(t[2])

    _print_stats_table("attn_output (forward)", [row("cpu"), row("cuda"), row("flagos")])

    if not args.no_bwd:
        def row_b(backend: str) -> tuple[str, float, float, float]:
            p = results.get(backend, {})
            t = p.get("bwd") or list(_nan_triple())
            return backend, float(t[0]), float(t[1]), float(t[2])

        _print_stats_table("hidden_states.grad (backward)", [row_b("cpu"), row_b("cuda"), row_b("flagos")])

    ok_all = all(results.get(b, {}).get("ok") for b in backends)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
