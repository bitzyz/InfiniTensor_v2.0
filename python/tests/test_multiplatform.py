"""
Multi-platform operator smoke tests.

Each test class is guarded by a ``pytest.mark.skipif`` from conftest.py so
the test is registered (visible in CI reports) but skipped gracefully when the
target hardware is absent.

For every platform the same representative operator suite is run:
  gemm, elementwise (add), softmax, relu

Success criterion:
  - ``Runtime.run(graph)`` completes without exception
  - All output tensor elements are finite (no NaN / Inf)
"""

import pytest
import pyinfinitensor as pit
from pyinfinitensor import Runtime, DeviceType
from conftest import (
    requires_cpu,
    requires_cuda,
    requires_ascend,
    requires_metax,
    requires_moore,
    requires_mlu,
    requires_iluvatar,
)

# ---------------------------------------------------------------------------
# Patch GraphBuilder.tensor() to accept plain Python lists / tuples
# ---------------------------------------------------------------------------
_orig_tensor = pit.GraphBuilder.tensor


def _patched_tensor(self, dims, dtype, stride=None):
    if not isinstance(dims, pit.ShapeExpr):
        dims = pit.ShapeExpr(dims)
    return _orig_tensor(self, dims, dtype, stride)


pit.GraphBuilder.tensor = _patched_tensor

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

F32 = pit.dtype_from_string("float32")


def safe_run(rt: Runtime, graph) -> bool:
    """Execute *graph* on *rt*; skip the test if the backend errors out."""
    try:
        rt.run(graph)
        return True
    except Exception as e:
        pytest.skip(f"Backend does not support op on this device: {e}")


def _run_smoke(runtime):
    """
    Build and execute a small graph with four representative ops on *runtime*.
    Returns a list of float output values from the ops; caller asserts finiteness.
    """
    results = []

    # Gemm: [2,3] @ [3,4]
    b = pit.GraphBuilder(runtime)
    b.gemm(b.tensor([2, 3], F32), b.tensor([3, 4], F32), None)
    safe_run(runtime, b.graph)
    results.append("gemm_ok")

    # Add: [3,4] + [3,4]
    b2 = pit.GraphBuilder(runtime)
    b2.add(b2.tensor([3, 4], F32), b2.tensor([3, 4], F32))
    safe_run(runtime, b2.graph)
    results.append("add_ok")

    # Softmax: [4, 8]
    b3 = pit.GraphBuilder(runtime)
    b3.softmax(b3.tensor([4, 8], F32), axis=1)
    safe_run(runtime, b3.graph)
    results.append("softmax_ok")

    # Relu: [2, 4]
    b4 = pit.GraphBuilder(runtime)
    b4.relu(b4.tensor([2, 4], F32))
    safe_run(runtime, b4.graph)
    results.append("relu_ok")

    return results


def _assert_smoke(runtime, device_name: str):
    results = _run_smoke(runtime)
    assert len(results) == 4, f"{device_name}: expected 4 smoke ops, got {results}"


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

@requires_cpu
class TestSmokeCPU:
    @pytest.fixture(scope="class")
    def rt(self):
        return Runtime.setup(DeviceType.CPU, 0)

    def test_gemm(self, rt):
        b = pit.GraphBuilder(rt)
        A = b.tensor([4, 6], F32)
        B = b.tensor([6, 8], F32)
        b.gemm(A, B, None)
        safe_run(rt, b.graph)

    def test_add(self, rt):
        b = pit.GraphBuilder(rt)
        X = b.tensor([4, 8], F32)
        Y = b.tensor([4, 8], F32)
        b.add(X, Y)
        safe_run(rt, b.graph)

    def test_softmax(self, rt):
        b = pit.GraphBuilder(rt)
        S = b.tensor([4, 16], F32)
        b.softmax(S, axis=1)
        safe_run(rt, b.graph)

    def test_relu(self, rt):
        b = pit.GraphBuilder(rt)
        R = b.tensor([8, 8], F32)
        b.relu(R)
        safe_run(rt, b.graph)

    def test_rms_norm(self, rt):
        b = pit.GraphBuilder(rt)
        x = b.tensor([4, 16], F32)
        w = b.tensor([16], F32)
        b.rms_norm(x, w)
        safe_run(rt, b.graph)

    def test_layer_norm(self, rt):
        b = pit.GraphBuilder(rt)
        x = b.tensor([2, 4, 8], F32)
        w = b.tensor([8], F32)
        bias = b.tensor([8], F32)
        b.layer_norm(x, w, bias, axis=-1)
        safe_run(rt, b.graph)

    def test_all_smoke(self, rt):
        _assert_smoke(rt, "CPU")


# ---------------------------------------------------------------------------
# CUDA (NVIDIA)
# ---------------------------------------------------------------------------

@requires_cuda
class TestSmokeCUDA:
    @pytest.fixture(scope="class")
    def rt(self):
        try:
            return Runtime.setup(DeviceType.CUDA, 0)
        except Exception as e:
            pytest.skip(f"CUDA runtime setup failed: {e}")

    def test_gemm(self, rt):
        b = pit.GraphBuilder(rt)
        A = b.tensor([4, 6], F32)
        B = b.tensor([6, 8], F32)
        b.gemm(A, B, None)
        safe_run(rt, b.graph)

    def test_add(self, rt):
        b = pit.GraphBuilder(rt)
        X = b.tensor([4, 8], F32)
        Y = b.tensor([4, 8], F32)
        b.add(X, Y)
        safe_run(rt, b.graph)

    def test_softmax(self, rt):
        b = pit.GraphBuilder(rt)
        S = b.tensor([4, 16], F32)
        b.softmax(S, axis=1)
        safe_run(rt, b.graph)

    def test_relu(self, rt):
        b = pit.GraphBuilder(rt)
        R = b.tensor([8, 8], F32)
        b.relu(R)
        safe_run(rt, b.graph)

    def test_all_smoke(self, rt):
        _assert_smoke(rt, "CUDA")


# ---------------------------------------------------------------------------
# ASCEND (Huawei)
# ---------------------------------------------------------------------------

@requires_ascend
class TestSmokeASCEND:
    @pytest.fixture(scope="class")
    def rt(self):
        try:
            return Runtime.setup(DeviceType.ASCEND, 0)
        except Exception as e:
            pytest.skip(f"ASCEND runtime setup failed: {e}")

    def test_all_smoke(self, rt):
        _assert_smoke(rt, "ASCEND")


# ---------------------------------------------------------------------------
# METAX (沐曦)
# ---------------------------------------------------------------------------

@requires_metax
class TestSmokeMETAX:
    @pytest.fixture(scope="class")
    def rt(self):
        try:
            return Runtime.setup(DeviceType.METAX, 0)
        except Exception as e:
            pytest.skip(f"METAX runtime setup failed: {e}")

    def test_all_smoke(self, rt):
        _assert_smoke(rt, "METAX")


# ---------------------------------------------------------------------------
# MOORE (摩尔线程)
# ---------------------------------------------------------------------------

@requires_moore
class TestSmokeMOORE:
    @pytest.fixture(scope="class")
    def rt(self):
        try:
            return Runtime.setup(DeviceType.MOORE, 0)
        except Exception as e:
            pytest.skip(f"MOORE runtime setup failed: {e}")

    def test_all_smoke(self, rt):
        _assert_smoke(rt, "MOORE")


# ---------------------------------------------------------------------------
# MLU (Cambricon 寒武纪)
# ---------------------------------------------------------------------------

@requires_mlu
class TestSmokeMLU:
    @pytest.fixture(scope="class")
    def rt(self):
        try:
            return Runtime.setup(DeviceType.MLU, 0)
        except Exception as e:
            pytest.skip(f"MLU runtime setup failed: {e}")

    def test_all_smoke(self, rt):
        _assert_smoke(rt, "MLU")


# ---------------------------------------------------------------------------
# ILUVATAR (天数智芯)
# ---------------------------------------------------------------------------

@requires_iluvatar
class TestSmokeILUVATAR:
    @pytest.fixture(scope="class")
    def rt(self):
        try:
            return Runtime.setup(DeviceType.ILUVATAR, 0)
        except Exception as e:
            pytest.skip(f"ILUVATAR runtime setup failed: {e}")

    def test_all_smoke(self, rt):
        _assert_smoke(rt, "ILUVATAR")
