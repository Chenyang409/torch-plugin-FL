"""
Basic Operations and Tensor Tests

Runs on both CUDA and flagos (MACA) devices via the --device argument.
torch_flagos makes flagos a drop-in replacement for cuda, so the test
logic is identical for both platforms.

Usage:
    pytest tests/integration/test_ops.py -v                          # default: cuda
    pytest tests/integration/test_ops.py -v --device flagos          # MACA platform
    pytest tests/integration/test_ops.py -v --device cuda            # CUDA platform
"""

import pytest
import torch


@pytest.fixture(scope="session")
def device(request):
    dev = request.config.getoption("--device")
    if dev == "flagos":
        import torch_flagos  # noqa: F401 — must be imported before torch on MACA

        if not torch_flagos.flagos.is_available():
            pytest.exit("flagos device is not available.")
        print(
            f"\nflagos device count={torch_flagos.flagos.device_count()}  "
            f"FlagGems enabled={torch_flagos.is_flaggems_enabled()}  "
            f"registered ops={len(torch_flagos.get_registered_ops())}"
        )
    else:
        if not torch.cuda.is_available():
            pytest.exit("CUDA is not available.")
        print(
            f"\nCUDA device: {torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}"
        )
    return f"{dev}:0"


# ---------------------------------------------------------------------------
# 1. Tensor creation
# ---------------------------------------------------------------------------


class TestTensorCreation:
    def test_randn(self, device):
        x = torch.randn(1024, 1024, device=device)
        assert x.device.type == device.split(":")[0]

    def test_zeros(self, device):
        assert torch.zeros(64, 64, device=device).sum().item() == 0.0

    def test_ones(self, device):
        assert torch.ones(64, 64, device=device).sum().item() == 64 * 64

    def test_arange(self, device):
        assert torch.arange(10, device=device)[-1].item() == 9

    def test_full(self, device):
        assert torch.full((4, 4), 3.14, device=device)[0, 0].item() == pytest.approx(
            3.14
        )


# ---------------------------------------------------------------------------
# 2. Arithmetic operations
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_add(self, device):
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        assert torch.allclose(a + b, torch.tensor([5.0, 7.0, 9.0], device=device))

    def test_sub(self, device):
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        assert torch.allclose(a - b, torch.tensor([-3.0, -3.0, -3.0], device=device))

    def test_mul(self, device):
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        assert torch.allclose(a * b, torch.tensor([4.0, 10.0, 18.0], device=device))

    def test_div(self, device):
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        assert torch.allclose(a / b, torch.tensor([0.25, 0.4, 0.5], device=device))

    def test_neg(self, device):
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        assert torch.allclose(-a, torch.tensor([-1.0, -2.0, -3.0], device=device))


# ---------------------------------------------------------------------------
# 3. Matrix operations
# ---------------------------------------------------------------------------


class TestMatrixOps:
    def test_mm(self, device):
        m, n = (
            torch.randn(256, 256, device=device),
            torch.randn(256, 256, device=device),
        )
        result = torch.mm(m, n)
        assert result.shape == (256, 256)
        assert result.device.type == device.split(":")[0]

    def test_bmm(self, device):
        a = torch.randn(8, 64, 128, device=device)
        b = torch.randn(8, 128, 64, device=device)
        assert torch.bmm(a, b).shape == (8, 64, 64)

    def test_matmul(self, device):
        m, n = (
            torch.randn(256, 256, device=device),
            torch.randn(256, 256, device=device),
        )
        assert torch.matmul(m, n).shape == (256, 256)


# ---------------------------------------------------------------------------
# 4. Reduction operations
# ---------------------------------------------------------------------------


class TestReductions:
    def test_sum(self, device):
        assert torch.randn(128, 128, device=device).sum().shape == torch.Size([])

    def test_mean(self, device):
        assert torch.randn(128, 128, device=device).mean().shape == torch.Size([])

    def test_max(self, device):
        assert torch.randn(128, 128, device=device).max().shape == torch.Size([])

    def test_min(self, device):
        assert torch.randn(128, 128, device=device).min().shape == torch.Size([])

    def test_std(self, device):
        assert torch.randn(128, 128, device=device).std().shape == torch.Size([])

    def test_norm(self, device):
        assert torch.randn(128, 128, device=device).norm().shape == torch.Size([])


# ---------------------------------------------------------------------------
# 5. Activation functions
# ---------------------------------------------------------------------------


class TestActivations:
    def test_relu(self, device):
        assert (
            torch.nn.functional.relu(torch.randn(64, 64, device=device)).min().item()
            >= 0.0
        )

    def test_sigmoid(self, device):
        s = torch.sigmoid(torch.randn(64, 64, device=device))
        assert (s > 0).all() and (s < 1).all()

    def test_tanh(self, device):
        t = torch.tanh(torch.randn(64, 64, device=device))
        assert (t > -1).all() and (t < 1).all()

    def test_softmax(self, device):
        sm = torch.softmax(torch.randn(64, 64, device=device), dim=-1)
        assert torch.allclose(sm.sum(dim=-1), torch.ones(64, device=device), atol=1e-4)

    def test_log_softmax(self, device):
        assert (
            torch.log_softmax(torch.randn(64, 64, device=device), dim=-1) <= 0
        ).all()

    def test_silu(self, device):
        x = torch.randn(64, 64, device=device)
        assert torch.nn.functional.silu(x).shape == x.shape

    def test_gelu(self, device):
        x = torch.randn(64, 64, device=device)
        assert torch.nn.functional.gelu(x).shape == x.shape


# ---------------------------------------------------------------------------
# 6. Shape operations
# ---------------------------------------------------------------------------


class TestShapeOps:
    def test_reshape(self, device):
        assert torch.randn(2, 3, 4, device=device).reshape(6, 4).shape == (6, 4)

    def test_view(self, device):
        assert torch.randn(2, 3, 4, device=device).view(24).shape == (24,)

    def test_transpose(self, device):
        assert torch.randn(2, 3, 4, device=device).transpose(0, 1).shape == (3, 2, 4)

    def test_permute(self, device):
        assert torch.randn(2, 3, 4, device=device).permute(2, 0, 1).shape == (4, 2, 3)

    def test_squeeze(self, device):
        assert torch.randn(1, 3, 1, device=device).squeeze().shape == (3,)

    def test_unsqueeze(self, device):
        assert torch.randn(2, 3, 4, device=device).unsqueeze(0).shape == (1, 2, 3, 4)

    def test_cat(self, device):
        t = torch.randn(2, 3, 4, device=device)
        assert torch.cat([t, t], dim=0).shape == (4, 3, 4)

    def test_stack(self, device):
        t = torch.randn(2, 3, 4, device=device)
        assert torch.stack([t, t], dim=0).shape == (2, 2, 3, 4)

    def test_split(self, device):
        assert len(torch.split(torch.randn(2, 3, 4, device=device), 1, dim=0)) == 2

    def test_chunk(self, device):
        assert len(torch.chunk(torch.randn(2, 3, 4, device=device), 2, dim=0)) == 2


# ---------------------------------------------------------------------------
# 7. Indexing and slicing
# ---------------------------------------------------------------------------


class TestIndexing:
    def test_slice_rows(self, device):
        t = torch.arange(24, device=device).reshape(4, 6).float()
        assert t[1:3].shape == (2, 6)

    def test_slice_cols(self, device):
        t = torch.arange(24, device=device).reshape(4, 6).float()
        assert t[:, 2:5].shape == (4, 3)

    def test_gather(self, device):
        t = torch.arange(24, device=device).reshape(4, 6).float()
        idx = torch.zeros(4, 1, dtype=torch.long, device=device)
        assert torch.gather(t, 1, idx).shape == (4, 1)

    def test_index_select(self, device):
        t = torch.arange(24, device=device).reshape(4, 6).float()
        idx = torch.tensor([0, 2], device=device)
        assert torch.index_select(t, 0, idx).shape == (2, 6)


# ---------------------------------------------------------------------------
# 8. Type casting
# ---------------------------------------------------------------------------


class TestTypeCasting:
    def test_to_float16(self, device):
        assert torch.randn(16, device=device).half().dtype == torch.float16

    def test_to_bfloat16(self, device):
        assert torch.randn(16, device=device).bfloat16().dtype == torch.bfloat16

    def test_to_int32(self, device):
        assert torch.randn(16, device=device).int().dtype == torch.int32

    def test_float16_to_float32(self, device):
        assert torch.randn(16, device=device).half().float().dtype == torch.float32


# ---------------------------------------------------------------------------
# 9. Device transfer
# ---------------------------------------------------------------------------


class TestDeviceTransfer:
    def test_roundtrip(self, device):
        cpu_t = torch.randn(32, 32)
        assert torch.allclose(cpu_t, cpu_t.to(device).cpu())

    def test_not_cpu(self, device):
        assert not torch.randn(4, device=device).is_cpu


# ---------------------------------------------------------------------------
# 10. Autograd
# ---------------------------------------------------------------------------


class TestAutograd:
    def test_grad_computed(self, device):
        x = torch.randn(4, 4, device=device, requires_grad=True)
        (x * x).sum().backward()
        assert x.grad is not None

    def test_grad_shape(self, device):
        x = torch.randn(4, 4, device=device, requires_grad=True)
        (x * x).sum().backward()
        assert x.grad.shape == (4, 4)

    def test_grad_values(self, device):
        x = torch.randn(4, 4, device=device, requires_grad=True)
        (x * x).sum().backward()
        assert torch.allclose(x.grad, 2 * x.detach(), atol=1e-5)


# ---------------------------------------------------------------------------
# 11. Synchronization
# ---------------------------------------------------------------------------


class TestSync:
    def test_synchronize(self, device):
        dev_type = device.split(":")[0]
        if dev_type == "flagos":
            import torch_flagos

            torch_flagos.flagos.synchronize()
        else:
            torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 12. Scaled dot-product attention
# ---------------------------------------------------------------------------


class TestSDPA:
    """Covers torch.nn.functional.scaled_dot_product_attention.

    On CUDA, PyTorch dispatches to a flash/efficient kernel. On flagos
    (PrivateUse1), it dispatches to aten._scaled_dot_product_flash_attention_for_cpu,
    which we decompose to matmul+softmax in torch_flagos._register_composite_ops.
    Same high-level call exercises both paths.
    """

    @staticmethod
    def _qkv(device, B=2, H=4, S=16, D=32, dtype=torch.float32):
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        return q, k, v

    def test_forward_runs(self, device):
        q, k, v = self._qkv(device)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert out.shape == q.shape
        assert out.device.type == device.split(":")[0]

    def test_matches_cpu_reference(self, device):
        q, k, v = self._qkv(device)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        q_cpu, k_cpu, v_cpu = q.cpu(), k.cpu(), v.cpu()
        ref = torch.nn.functional.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
        assert torch.allclose(out.cpu(), ref, atol=1e-4, rtol=1e-4)

    def test_causal(self, device):
        q, k, v = self._qkv(device)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.cpu(), k.cpu(), v.cpu(), is_causal=True
        )
        assert torch.allclose(out.cpu(), ref, atol=1e-4, rtol=1e-4)
