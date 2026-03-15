import pytest
import torch
import torch.nn as nn
import numpy as np
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType


def _build_input_tensors(input_info):
    return [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]


def _run_once(model, runtime, input_info):
    input_tensors = _build_input_tensors(input_info)
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)
    translator.run(input_tensors)
    outputs = translator.get_outputs()
    assert len(outputs) == 1
    return outputs[0]


def _run_dynamic(model, runtime, import_input_info, run_input_infos):
    input_tensors = _build_input_tensors(import_input_info)
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    output_shapes = []
    for input_info in run_input_infos:
        run_inputs = _build_input_tensors(input_info)
        translator.run(run_inputs)
        outputs = translator.get_outputs()
        assert len(outputs) == 1
        output_shapes.append(tuple(outputs[0].shape))
    return output_shapes


class _BinaryModel(nn.Module):
    def __init__(self, op_name):
        super().__init__()
        self.op_name = op_name

    def forward(self, x, y):
        if self.op_name == "add":
            return x + y
        if self.op_name == "sub":
            return x - y
        if self.op_name == "mul":
            return x * y
        raise ValueError(f"Unsupported binary op: {self.op_name}")


class _UnaryModel(nn.Module):
    def __init__(self, op_name):
        super().__init__()
        self.op_name = op_name

    def forward(self, x):
        if self.op_name == "relu":
            return torch.relu(x)
        if self.op_name == "sigmoid":
            return torch.sigmoid(x)
        if self.op_name == "silu":
            return torch.nn.functional.silu(x)
        if self.op_name == "gelu":
            return torch.nn.functional.gelu(x)
        if self.op_name == "tanh":
            return torch.tanh(x)
        if self.op_name == "softplus":
            return torch.nn.functional.softplus(x)
        raise ValueError(f"Unsupported unary op: {self.op_name}")


def test_basic_matmul(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class MatmulModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    model = MatmulModel()
    # Randomly initialize inputs, passed shapes can differ from actual values, but data types must match
    input_info = [((5, 4), "float32"), ((4, 3), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)
    # Run
    translator.run(input_tensors)

    # Get outputs
    outputs = translator.get_outputs()

    # Verify
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 5, 3)
    print("Test passed")


def test_dynamic_matmul(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class MatmulModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    model = MatmulModel()
    # Randomly initialize inputs, passed shapes can differ from actual values, but data types must match
    input_info = [((5, 4), "float32"), ((4, 7), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    input_info_1 = [((15, 4), "float32"), ((4, 12), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (1, 15, 12)

    input_info_2 = [((3, 20), "float32"), ((20, 10), "float32")]
    input_tensors_2 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_2
    ]
    translator.run(input_tensors_2)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (1, 3, 10)
    print("Test passed")


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
def test_basic_binary_ops(runtime, torch_rng_seed, op_name):
    print(f"Testing basic {op_name} with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    out = _run_once(
        _BinaryModel(op_name),
        runtime,
        [((5, 4), "float32"), ((5, 4), "float32")],
    )
    assert tuple(out.shape) == (5, 4)


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
def test_dynamic_binary_ops(runtime, torch_rng_seed, op_name):
    print(f"Testing dynamic {op_name} with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    shapes = _run_dynamic(
        _BinaryModel(op_name),
        runtime,
        [((5, 4), "float32"), ((5, 4), "float32")],
        [
            [((7, 3), "float32"), ((7, 3), "float32")],
            [((2, 9), "float32"), ((2, 9), "float32")],
        ],
    )
    assert shapes == [(7, 3), (2, 9)]


@pytest.mark.parametrize(
    "op_name", ["relu", "sigmoid", "silu", "gelu", "tanh", "softplus"]
)
def test_basic_unary_ops(runtime, torch_rng_seed, op_name):
    print(f"Testing basic {op_name} with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    out = _run_once(_UnaryModel(op_name), runtime, [((3, 5, 4), "float32")])
    assert tuple(out.shape) == (3, 5, 4)


@pytest.mark.parametrize(
    "op_name", ["relu", "sigmoid", "silu", "gelu", "tanh", "softplus"]
)
def test_dynamic_unary_ops(runtime, torch_rng_seed, op_name):
    print(f"Testing dynamic {op_name} with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    shapes = _run_dynamic(
        _UnaryModel(op_name),
        runtime,
        [((3, 5, 4), "float32")],
        [
            [((2, 7, 4), "float32")],
            [((1, 9, 4), "float32")],
        ],
    )
    assert shapes == [(2, 7, 4), (1, 9, 4)]


def test_basic_softmax(runtime, torch_rng_seed):
    print(f"Testing basic softmax with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class SoftmaxModel(nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=-1)

    with pytest.raises(RuntimeError):
        _run_once(SoftmaxModel(), runtime, [((3, 6, 8), "float32")])


def test_dynamic_softmax(runtime, torch_rng_seed):
    print(f"Testing dynamic softmax with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class SoftmaxModel(nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=-1)

    with pytest.raises(RuntimeError):
        _run_dynamic(
            SoftmaxModel(),
            runtime,
            [((3, 6, 8), "float32")],
            [
                [((2, 7, 8), "float32")],
                [((1, 9, 8), "float32")],
            ],
        )


def test_basic_log_softmax(runtime, torch_rng_seed):
    print(f"Testing basic log_softmax with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class LogSoftmaxModel(nn.Module):
        def forward(self, x):
            return torch.log_softmax(x, dim=1)

    out = _run_once(LogSoftmaxModel(), runtime, [((3, 6, 8), "float32")])
    assert tuple(out.shape) == (3, 6, 8)


def test_dynamic_log_softmax(runtime, torch_rng_seed):
    print(f"Testing dynamic log_softmax with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class LogSoftmaxModel(nn.Module):
        def forward(self, x):
            return torch.log_softmax(x, dim=1)

    shapes = _run_dynamic(
        LogSoftmaxModel(),
        runtime,
        [((3, 6, 8), "float32")],
        [
            [((2, 7, 8), "float32")],
            [((1, 9, 8), "float32")],
        ],
    )
    assert shapes == [(2, 7, 8), (1, 9, 8)]


def test_basic_conv2d(runtime, torch_rng_seed):
    print(f"Testing basic conv2d with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class Conv2dModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, dilation=1)

        def forward(self, x):
            return self.conv(x)

    with pytest.raises(RuntimeError):
        _run_once(Conv2dModel(), runtime, [((2, 3, 16, 20), "float32")])


def test_dynamic_conv2d(runtime, torch_rng_seed):
    print(f"Testing dynamic conv2d with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class Conv2dModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, dilation=1)

        def forward(self, x):
            return self.conv(x)

    with pytest.raises(RuntimeError):
        _run_dynamic(
            Conv2dModel(),
            runtime,
            [((2, 3, 16, 20), "float32")],
            [
                [((1, 3, 10, 14), "float32")],
                [((3, 3, 12, 18), "float32")],
            ],
        )


def test_basic_layer_norm(runtime, torch_rng_seed):
    print(f"Testing basic layer_norm with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class LayerNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(8, eps=1e-5)

        def forward(self, x):
            return self.norm(x)

    out = _run_once(LayerNormModel(), runtime, [((2, 6, 8), "float32")])
    assert tuple(out.shape) == (2, 6, 8)


def test_dynamic_layer_norm(runtime, torch_rng_seed):
    print(f"Testing dynamic layer_norm with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class LayerNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(8, eps=1e-5)

        def forward(self, x):
            return self.norm(x)

    shapes = _run_dynamic(
        LayerNormModel(),
        runtime,
        [((2, 6, 8), "float32")],
        [
            [((1, 10, 8), "float32")],
            [((3, 4, 8), "float32")],
        ],
    )
    assert shapes == [(1, 10, 8), (3, 4, 8)]


@pytest.mark.skipif(not hasattr(nn, "RMSNorm"), reason="torch.nn.RMSNorm not available")
def test_basic_rms_norm(runtime, torch_rng_seed):
    print(f"Testing basic rms_norm with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class RMSNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.RMSNorm(8, eps=1e-6)

        def forward(self, x):
            return self.norm(x)

    out = _run_once(RMSNormModel(), runtime, [((2, 6, 8), "float32")])
    assert tuple(out.shape) == (2, 6, 8)


@pytest.mark.skipif(not hasattr(nn, "RMSNorm"), reason="torch.nn.RMSNorm not available")
def test_dynamic_rms_norm(runtime, torch_rng_seed):
    print(f"Testing dynamic rms_norm with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class RMSNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.RMSNorm(8, eps=1e-6)

        def forward(self, x):
            return self.norm(x)

    shapes = _run_dynamic(
        RMSNormModel(),
        runtime,
        [((2, 6, 8), "float32")],
        [
            [((1, 10, 8), "float32")],
            [((3, 4, 8), "float32")],
        ],
    )
    assert shapes == [(1, 10, 8), (3, 4, 8)]


def test_basic_linalg_vector_norm(runtime, torch_rng_seed):
    print(f"Testing basic linalg_vector_norm with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class VectorNormModel(nn.Module):
        def forward(self, x):
            return torch.linalg.vector_norm(x, ord=2, dim=1)

    with pytest.raises(RuntimeError):
        _run_once(VectorNormModel(), runtime, [((5, 4, 3), "float32")])


def test_dynamic_linalg_vector_norm(runtime, torch_rng_seed):
    print(f"Testing dynamic linalg_vector_norm with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class VectorNormModel(nn.Module):
        def forward(self, x):
            return torch.linalg.vector_norm(x, ord=2, dim=1)

    with pytest.raises(RuntimeError):
        _run_dynamic(
            VectorNormModel(),
            runtime,
            [((5, 4, 3), "float32")],
            [
                [((2, 7, 3), "float32")],
                [((4, 6, 3), "float32")],
            ],
        )


def test_basic_linalg_norm(runtime, torch_rng_seed):
    print(f"Testing basic linalg_norm with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class NormModel(nn.Module):
        def forward(self, x):
            return torch.linalg.norm(x, ord=2, dim=1)

    with pytest.raises(RuntimeError):
        _run_once(NormModel(), runtime, [((5, 4, 3), "float32")])


def test_dynamic_linalg_norm(runtime, torch_rng_seed):
    print(f"Testing dynamic linalg_norm with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class NormModel(nn.Module):
        def forward(self, x):
            return torch.linalg.norm(x, ord=2, dim=1)

    with pytest.raises(RuntimeError):
        _run_dynamic(
            NormModel(),
            runtime,
            [((5, 4, 3), "float32")],
            [
                [((2, 7, 3), "float32")],
                [((4, 6, 3), "float32")],
            ],
        )


def test_basic_clamp(runtime, torch_rng_seed):
    print(f"Testing basic clamp with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class ClampModel(nn.Module):
        def forward(self, x):
            return torch.clamp(x, min=-0.5, max=0.5)

    out = _run_once(ClampModel(), runtime, [((3, 5, 4), "float32")])
    assert tuple(out.shape) == (3, 5, 4)


def test_dynamic_clamp(runtime, torch_rng_seed):
    print(f"Testing dynamic clamp with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class ClampModel(nn.Module):
        def forward(self, x):
            return torch.clamp(x, min=-0.5, max=0.5)

    shapes = _run_dynamic(
        ClampModel(),
        runtime,
        [((3, 5, 4), "float32")],
        [
            [((2, 7, 4), "float32")],
            [((1, 9, 4), "float32")],
        ],
    )
    assert shapes == [(2, 7, 4), (1, 9, 4)]


def test_basic_clip(runtime, torch_rng_seed):
    print(f"Testing basic clip with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class ClipModel(nn.Module):
        def forward(self, x):
            return torch.clip(x, min=-0.25, max=0.25)

    out = _run_once(ClipModel(), runtime, [((3, 5, 4), "float32")])
    assert tuple(out.shape) == (3, 5, 4)


def test_dynamic_clip(runtime, torch_rng_seed):
    print(f"Testing dynamic clip with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class ClipModel(nn.Module):
        def forward(self, x):
            return torch.clip(x, min=-0.25, max=0.25)

    shapes = _run_dynamic(
        ClipModel(),
        runtime,
        [((3, 5, 4), "float32")],
        [
            [((2, 7, 4), "float32")],
            [((1, 9, 4), "float32")],
        ],
    )
    assert shapes == [(2, 7, 4), (1, 9, 4)]


if __name__ == "__main__":
    # Can run this file directly
    import sys

    # Run all tests using pytest
    exit_code = pytest.main(
        [
            __file__,
            "-v",  # Verbose output
            "-s",  # Show print output
            "--tb=short",  # Simplified error traceback
        ]
    )

    sys.exit(0 if exit_code == 0 else 1)
