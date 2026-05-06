#!/usr/bin/env python3
"""
Compare forward outputs of a submodule on CUDA vs FlagOS using tensors saved by
test_qwen3_train.py (first_forward_nonfinite_*.pt).

The dump does not contain module weights; this script loads the same HuggingFace
checkpoint, takes ``get_submodule(module_name)``, deep-copies it onto each device,
runs forward with saved args/kwargs, and reports max |out_cuda - out_flagos|.

Usage:
  python compare_dump_cuda_flagos.py \\
    --dump debug_dump/first_forward_nonfinite_step2_model_embed_tokens.pt \\
    --model /path/to/Qwen3-0.6B

  # Only CUDA (no torch_flagos), sanity-check vs saved reference output:
  python compare_dump_cuda_flagos.py --dump ... --model ... --cuda-only
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Any

import torch


def _move_structure(obj: Any, device: str | torch.device):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            return obj.clone().detach().to(device=device, dtype=obj.dtype)
        return obj.clone().detach().to(device=device)
    if isinstance(obj, tuple):
        return tuple(_move_structure(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_structure(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_structure(v, device) for k, v in obj.items()}
    return obj


def _tensor_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d = (a.float() - b.float()).abs()
    return {
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
    }


def _compare_tensors(
    a: torch.Tensor | None, b: torch.Tensor | None, label: str, rtol: float, atol: float
) -> bool:
    if a is None and b is None:
        print(f"  [{label}] both None — skip")
        return True
    if a is None or b is None:
        print(f"  [{label}] mismatch: one is None")
        return False
    if a.shape != b.shape:
        print(f"  [{label}] shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    af, bf = a.detach().float().cpu(), b.detach().float().cpu()
    st = _tensor_stats(af, bf)
    ok = torch.allclose(af, bf, rtol=rtol, atol=atol)
    flag = "OK" if ok else "DIFF"
    print(
        f"  [{label}] {flag} max_abs={st['max_abs']:.6e} mean_abs={st['mean_abs']:.6e} "
        f"rtol={rtol} atol={atol}"
    )
    return ok


def _compare_structure(a: Any, b: Any, label: str, rtol: float, atol: float) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return _compare_tensors(a, b, label, rtol, atol)
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        ok = True
        for i, (x, y) in enumerate(zip(a, b)):
            ok = _compare_structure(x, y, f"{label}[{i}]", rtol, atol) and ok
        return ok
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        ok = True
        for i, (x, y) in enumerate(zip(a, b)):
            ok = _compare_structure(x, y, f"{label}[{i}]", rtol, atol) and ok
        return ok
    print(f"  [{label}] unsupported or mismatched structure: {type(a)} vs {type(b)}")
    return False


def _call_forward(module: torch.nn.Module, args: tuple, kwargs: dict) -> Any:
    if args and kwargs:
        return module(*args, **kwargs)
    if args:
        return module(*args)
    if kwargs:
        return module(**kwargs)
    return module()


def main() -> int:
    parser = argparse.ArgumentParser(description="CUDA vs FlagOS output compare from dump .pt")
    parser.add_argument("--dump", required=True, help="Path to first_forward_nonfinite_*.pt")
    parser.add_argument(
        "--model",
        default="/nfs/hcr/models/Qwen/Qwen3-0.6B",
        help="Same pretrained path as training (for submodule weights)",
    )
    parser.add_argument("--cuda-only", action="store_true", help="Only run CUDA vs saved output")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    try:
        payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.dump, map_location="cpu")
    kind = payload.get("kind", "")
    if kind == "backward_first_nonfinite":
        print(
            "This file is a backward hook dump (grad_input/grad_output). "
            "This script only runs forward compares. Use a first_forward_nonfinite_*.pt file."
        )
        return 2
    if kind != "forward_first_nonfinite":
        print(f"Unknown kind={kind!r}; expected forward_first_nonfinite.")

    module_name = payload["module_name"]
    module_class = payload["module_class"]
    fwd_args: tuple = payload.get("forward_args") or ()
    fwd_kw: dict = payload.get("forward_kwargs") or {}
    ref_out = payload.get("output")

    if module_name in ("(root)", ""):
        print("Cannot compare root module from this script.")
        return 2

    print(f"Dump: {args.dump}")
    print(f"Submodule: `{module_name}` ({module_class})")

    from transformers import AutoModelForCausalLM

    load_kwargs = dict(torch_dtype=torch.float32, device_map="cpu")
    load_kwargs["attn_implementation"] = "eager"
    full = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    full.eval()

    try:
        sub_cpu = full.get_submodule(module_name)
    except AttributeError as e:
        print(f"get_submodule failed: {e}")
        print("Available prefixes (first 20):")
        for i, (n, _) in enumerate(full.named_modules()):
            if i >= 20:
                break
            print(f"  {n}")
        return 2

    sub_cuda = copy.deepcopy(sub_cpu).cuda().eval()
    args_cuda = _move_structure(fwd_args, "cuda")
    kw_cuda = _move_structure(fwd_kw, "cuda")

    with torch.no_grad():
        try:
            out_cuda = _call_forward(sub_cuda, args_cuda, kw_cuda)
        except Exception as e:
            print(f"CUDA forward failed: {e}")
            return 1

    all_ok = True

    if args.cuda_only:
        print("CUDA-only: comparing CUDA output to saved reference output.")
        all_ok = _compare_structure(out_cuda, ref_out, "cuda_vs_saved", args.rtol, args.atol)
        return 0 if all_ok else 1

    try:
        import torch_flagos
    except ImportError:
        print("torch_flagos not installed; use --cuda-only or install torch_flagos.")
        return 2

    if not torch_flagos.flagos.is_available():
        print("FlagOS not available on this machine.")
        return 2

    torch_flagos.flagos.set_device(0)
    sub_flag = copy.deepcopy(sub_cpu).to("flagos:0").eval()
    args_flag = _move_structure(fwd_args, "flagos:0")
    kw_flag = _move_structure(fwd_kw, "flagos:0")

    with torch.no_grad():
        try:
            out_flag = _call_forward(sub_flag, args_flag, kw_flag)
        except Exception as e:
            print(f"FlagOS forward failed: {e}")
            return 1

    print("Compare CUDA vs FlagOS (same weights from deepcopy of CPU submodule):")
    if torch.is_tensor(out_cuda) and torch.is_tensor(out_flag):
        all_ok = _compare_tensors(
            out_cuda.cpu(), out_flag.cpu(), "cuda_vs_flagos", args.rtol, args.atol
        )
    elif isinstance(out_cuda, tuple) and isinstance(out_flag, tuple):
        if len(out_cuda) != len(out_flag):
            print(f"tuple len mismatch {len(out_cuda)} vs {len(out_flag)}")
            return 1
        for i, (a, b) in enumerate(zip(out_cuda, out_flag)):
            if torch.is_tensor(a) and torch.is_tensor(b):
                all_ok = (
                    _compare_tensors(
                        a.cpu(), b.cpu(), f"cuda_vs_flagos[{i}]", args.rtol, args.atol
                    )
                    and all_ok
                )
            else:
                print(f"  [cuda_vs_flagos[{i}]] skip non-tensor {type(a)} {type(b)}")
    else:
        print(f"Unexpected output types: {type(out_cuda)} vs {type(out_flag)}")
        return 1

    if ref_out is not None:
        print("Optional: CUDA vs saved reference (weights at dump time may differ — often large diff):")
        _compare_structure(out_cuda, ref_out, "cuda_vs_saved", args.rtol, args.atol)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
