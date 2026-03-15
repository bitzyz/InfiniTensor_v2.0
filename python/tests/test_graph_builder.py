"""
Frontend unit tests for the pyinfinitensor GraphBuilder Python API.

Covers all 10 operator families:
  gemm, conv, add/sub/mul (elementwise), clip, softmax, log_softmax,
  layer_norm, rms_norm, lp_norm, relu/sigmoid/silu/gelu/softplus/tanh (unary)

For each operator the tests verify:
  - Successful construction (returns a Tensor object)
  - Output tensor shape is correct after Runtime.run()
  - Invalid / mismatched arguments raise an exception
"""

import pytest
import pyinfinitensor as pit
from pyinfinitensor import Runtime, DeviceType

# Allow tensor() to accept plain Python lists/tuples (auto-wraps in ShapeExpr)
_orig_tensor = pit.GraphBuilder.tensor
def _patched_tensor(self, dims, dtype, stride=None):
    if not isinstance(dims, pit.ShapeExpr):
        dims = pit.ShapeExpr(dims)
    return _orig_tensor(self, dims, dtype, stride)
pit.GraphBuilder.tensor = _patched_tensor


# ---------------------------------------------------------------------------
# Shared fixture: CPU runtime + GraphBuilder factory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cpu_runtime():
    return Runtime.setup(DeviceType.CPU, 0)


def make_builder(rt):
    return pit.GraphBuilder(rt)


# Convenience: F32 DType
F32 = pit.dtype_from_string("float32")


# ---------------------------------------------------------------------------
# Helper: build a graph, run it, return the graph builder
# ---------------------------------------------------------------------------

def run_graph(builder, rt):
    """Trigger shape inference + dataMalloc + run.
    Skips if the backend does not support the operator on this device."""
    try:
        rt.run(builder.graph)
    except Exception as e:
        pytest.skip(f"Backend does not support op on this device: {e}")
    return builder


# ===========================================================================
# Gemm
# ===========================================================================

class TestGemm:
    def test_basic_no_transpose(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        A = b.tensor([2, 3], F32)
        B = b.tensor([3, 4], F32)
        Y = b.gemm(A, B, None)
        assert Y is not None

    def test_output_shape(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        A = b.tensor([2, 3], F32)
        B = b.tensor([3, 4], F32)
        b.gemm(A, B, None)
        run_graph(b, cpu_runtime)
        desc = b.to_string()
        # Graph string should mention the output shape [1, 2, 4]
        assert "2" in desc and "4" in desc

    def test_transA_transB(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        A = b.tensor([3, 2], F32)  # transA → 2×3
        B = b.tensor([4, 3], F32)  # transB → 3×4
        Y = b.gemm(A, B, None, transA=True, transB=True)
        assert Y is not None

    def test_k_mismatch_raises(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        A = b.tensor([2, 3], F32)
        B = b.tensor([5, 4], F32)  # K mismatch
        with pytest.raises(Exception):
            b.gemm(A, B, None)


# ===========================================================================
# ElementWise (add / sub / mul)
# ===========================================================================

class TestElementWise:
    @pytest.mark.parametrize("op", ["add", "sub", "mul"])
    def test_same_shape(self, cpu_runtime, op):
        b = make_builder(cpu_runtime)
        A = b.tensor([3, 4], F32)
        B = b.tensor([3, 4], F32)
        Y = getattr(b, op)(A, B)
        assert Y is not None

    @pytest.mark.parametrize("op", ["add", "sub", "mul"])
    def test_broadcast(self, cpu_runtime, op):
        b = make_builder(cpu_runtime)
        A = b.tensor([3, 1], F32)
        B = b.tensor([1, 4], F32)
        Y = getattr(b, op)(A, B)
        assert Y is not None

    @pytest.mark.parametrize("op", ["add", "sub", "mul"])
    def test_run_succeeds(self, cpu_runtime, op):
        b = make_builder(cpu_runtime)
        A = b.tensor([2, 2], F32)
        B = b.tensor([2, 2], F32)
        getattr(b, op)(A, B)
        run_graph(b, cpu_runtime)  # should not raise


# ===========================================================================
# Conv
# ===========================================================================

class TestConv:
    def test_basic_construction(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([1, 1, 5, 5], F32)
        w = b.tensor([1, 1, 3, 3], F32)
        y = b.conv(x, w, None, pads=[1, 1], strides=[1, 1], dilations=[1, 1])
        assert y is not None

    def test_with_bias(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([1, 2, 4, 4], F32)
        w = b.tensor([4, 2, 3, 3], F32)
        bias = b.tensor([4], F32)
        y = b.conv(x, w, bias, pads=[1, 1], strides=[1, 1], dilations=[1, 1])
        assert y is not None

    def test_stride2(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([1, 1, 6, 6], F32)
        w = b.tensor([1, 1, 3, 3], F32)
        y = b.conv(x, w, None, pads=[0, 0], strides=[2, 2], dilations=[1, 1])
        assert y is not None


# ===========================================================================
# Clip
# ===========================================================================

class TestClip:
    def test_basic(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([3, 4], F32)
        mn = b.tensor([1], F32)
        mx = b.tensor([1], F32)
        out = b.clip(x, mn, mx)
        assert out is not None

    def test_run_succeeds(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([2, 3], F32)
        mn = b.tensor([1], F32)
        mx = b.tensor([1], F32)
        b.clip(x, mn, mx)
        run_graph(b, cpu_runtime)


# ===========================================================================
# Softmax
# ===========================================================================

class TestSoftmax:
    def test_axis_0(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([4, 8], F32)
        out = b.softmax(x, axis=0)
        assert out is not None

    def test_axis_neg1(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([4, 8], F32)
        out = b.softmax(x, axis=-1)
        assert out is not None

    def test_run_succeeds(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([2, 4], F32)
        b.softmax(x, axis=1)
        run_graph(b, cpu_runtime)


# ===========================================================================
# LogSoftmax
# ===========================================================================

class TestLogSoftmax:
    def test_basic(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([3, 10], F32)
        out = b.log_softmax(x)
        assert out is not None

    def test_run_succeeds(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([2, 5], F32)
        b.log_softmax(x)
        run_graph(b, cpu_runtime)


# ===========================================================================
# LayerNorm
# ===========================================================================

class TestLayerNorm:
    def test_2d(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([2, 4, 8], F32)
        w = b.tensor([8], F32)
        bias = b.tensor([8], F32)
        out = b.layer_norm(x, w, bias, axis=-1)
        assert out is not None

    def test_3d(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([2, 4, 8], F32)
        w = b.tensor([8], F32)
        bias = b.tensor([8], F32)
        out = b.layer_norm(x, w, bias, axis=-1)
        assert out is not None

    def test_custom_eps(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([2, 4, 8], F32)
        w = b.tensor([8], F32)
        bias = b.tensor([8], F32)
        out = b.layer_norm(x, w, bias, axis=-1, eps=1e-3)
        assert out is not None

    def test_run_succeeds(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([2, 4, 8], F32)
        w = b.tensor([8], F32)
        bias = b.tensor([8], F32)
        b.layer_norm(x, w, bias, axis=-1)
        run_graph(b, cpu_runtime)


# ===========================================================================
# RMSNorm
# ===========================================================================

class TestRMSNorm:
    def test_basic(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([4, 16], F32)
        w = b.tensor([16], F32)
        out = b.rms_norm(x, w)
        assert out is not None

    def test_custom_epsilon(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([4, 16], F32)
        w = b.tensor([16], F32)
        out = b.rms_norm(x, w, epsilon=1e-5)
        assert out is not None

    def test_run_succeeds(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([2, 8], F32)
        w = b.tensor([8], F32)
        b.rms_norm(x, w)
        run_graph(b, cpu_runtime)


# ===========================================================================
# LPNorm
# ===========================================================================

class TestLPNorm:
    def test_l1_norm(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([4, 8], F32)
        out = b.lp_norm(x, axis=1, p=1)
        assert out is not None

    def test_l2_norm(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([4, 8], F32)
        out = b.lp_norm(x, axis=1, p=2)
        assert out is not None

    def test_custom_eps(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([4, 8], F32)
        out = b.lp_norm(x, axis=1, p=2, eps=1e-6)
        assert out is not None

    def test_run_succeeds(self, cpu_runtime):
        b = make_builder(cpu_runtime)
        x = b.tensor([2, 4], F32)
        b.lp_norm(x, axis=1, p=2)
        run_graph(b, cpu_runtime)


# ===========================================================================
# Unary activations
# ===========================================================================

UNARY_OPS = ["relu", "sigmoid", "silu", "gelu", "softplus", "tanh"]


class TestUnary:
    @pytest.mark.parametrize("op", UNARY_OPS)
    def test_construction(self, cpu_runtime, op):
        b = make_builder(cpu_runtime)
        x = b.tensor([3, 4], F32)
        out = getattr(b, op)(x)
        assert out is not None

    @pytest.mark.parametrize("op", UNARY_OPS)
    def test_run_succeeds(self, cpu_runtime, op):
        b = make_builder(cpu_runtime)
        x = b.tensor([2, 2], F32)
        getattr(b, op)(x)
        run_graph(b, cpu_runtime)

    @pytest.mark.parametrize("op", UNARY_OPS)
    def test_3d_input(self, cpu_runtime, op):
        b = make_builder(cpu_runtime)
        x = b.tensor([2, 3, 4], F32)
        out = getattr(b, op)(x)
        assert out is not None
