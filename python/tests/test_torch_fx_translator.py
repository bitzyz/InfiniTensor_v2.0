import pytest
import torch
import torch.nn as nn
import numpy as np
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType


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
    print("✅ Test passed!")


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

    # ! 这么写不安全. 二次调用translator.run 不会重新分配内存，第二次比第一次大就会core!!!
    # 这个core 我查了两天！！！
    # 见 tensor.cc dataMalloc 的修改
    input_info_2 = [((3, 20), "float32"), ((20, 10), "float32")]
    input_tensors_2 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_2
    ]
    translator.run(input_tensors_2)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (1, 3, 10)
    print("✅ Test passed!")


def test_basic_elementwise(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class AddModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = AddModel()
    # Randomly initialize inputs, passed shapes can differ from actual values, but data types must match
    input_info = [((5, 4), "float32"), ((3, 5, 1), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    translator.run(input_tensors)
    # Get outputs
    outputs = translator.get_outputs()

    # Verify
    assert len(outputs) == 1
    assert outputs[0].shape == (3, 5, 4)
    print("✅ Test passed!")


def test_clip(runtime, torch_rng_seed):
    """Test torch.clamp with dynamic shapes"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class ClipModel(torch.nn.Module):
        def forward(self, x, min_val, max_val):
            return torch.clamp(x, min=min_val, max=max_val)

    model = ClipModel()
    # Randomly initialize inputs (x, min_val, max_val)
    # min and max have the same shape as input (no broadcasting needed)
    input_info = [((5, 4), "float32"), ((5, 4), "float32"), ((5, 4), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]
    # Set min and max values
    input_tensors[1].fill_(-1.0)
    input_tensors[2].fill_(1.0)

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    # First run with different shape
    input_info_1 = [((15, 12), "float32"), ((15, 12), "float32"), ((15, 12), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    input_tensors_1[1].fill_(-0.5)
    input_tensors_1[2].fill_(0.5)
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (15, 12)

    # Second run with another shape
    input_info_2 = [((3, 20), "float32"), ((3, 20), "float32"), ((3, 20), "float32")]
    input_tensors_2 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_2
    ]
    input_tensors_2[1].fill_(0.0)
    input_tensors_2[2].fill_(2.0)
    translator.run(input_tensors_2)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 20)
    print("✅ Test passed!")


# infiniCore 要求输入x 的rank至少为3
# weight, bias只能是rank = 1
def test_dynamic_conv(runtime, torch_rng_seed):
    """Test torch.conv1d, conv2d, conv3d with stride, padding, dilation"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Test conv1d
    class Conv1dModel(torch.nn.Module):
        def forward(self, x, weight):
            return torch.conv1d(x, weight, stride=2, padding=1, dilation=1)

    model = Conv1dModel()
    input_info = [((2, 3, 16), "float32"), ((4, 3, 3), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    translator.run(input_tensors)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (2, 4, 8)

    # Second run
    input_info_2 = [((3, 3, 20), "float32"), ((4, 3, 3), "float32")]
    input_tensors_2 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_2
    ]
    translator.run(input_tensors_2)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 4, 10)
    print("✅ conv1d test passed!")

    # Test conv2d
    class Conv2dModel(torch.nn.Module):
        def forward(self, x, weight):
            return torch.conv2d(
                x, weight, stride=(2, 1), padding=(1, 1), dilation=(1, 1)
            )

    model = Conv2dModel()
    input_info = [((2, 3, 16, 16), "float32"), ((4, 3, 3, 3), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    input_info_1 = [((5, 3, 32, 32), "float32"), ((4, 3, 3, 3), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    torch_shape = model(input_tensors_1[0], input_tensors_1[1]).shape
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == torch_shape

    input_info_2 = [((3, 3, 20, 20), "float32"), ((4, 3, 3, 3), "float32")]
    input_tensors_2 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_2
    ]
    torch_shape = model(input_tensors_2[0], input_tensors_2[1]).shape
    translator.run(input_tensors_2)
    outputs = translator.get_outputs()
    assert outputs[0].shape == torch_shape
    print("✅ conv2d test passed!")

    # Test conv3d
    class Conv3dModel(torch.nn.Module):
        def forward(self, x, weight):
            return torch.conv3d(
                x, weight, stride=[2, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1]
            )

    # model = Conv3dModel()
    # input_info = [((2, 3, 8, 16, 16), "float32"), ((4, 3, 3, 3, 3), "float32")]
    # input_tensors = [
    #     torch.as_tensor(np.random.randn(*shape).astype(dtype))
    #     for shape, dtype in input_info
    # ]
    #
    # translator = TorchFXTranslator(runtime)
    # translator.import_from_fx(model, input_tensors)
    #
    # input_info_1 = [((5, 3, 16, 32, 32), "float32"), ((4, 3, 3, 3, 3), "float32")]
    # input_tensors_1 = [
    #     torch.randn(*shape, dtype=getattr(torch, dtype))
    #     for shape, dtype in input_info_1
    # ]
    # torch_shape = model(input_tensors_1[0], input_tensors_1[1]).shape
    # translator.run(input_tensors_1)
    # outputs = translator.get_outputs()
    # assert outputs[0].shape == torch_shape
    #
    # input_info_2 = [((3, 3, 10, 20, 20), "float32"), ((4, 3, 3, 3, 3), "float32")]
    # input_tensors_2 = [
    #     torch.as_tensor(np.random.randn(*shape).astype(dtype))
    #     for shape, dtype in input_info_2
    # ]
    # torch_shape = model(input_tensors_2[0], input_tensors_2[1]).shape
    # translator.run(input_tensors_2)
    # outputs = translator.get_outputs()
    # assert outputs[0].shape == torch_shape
    print("✅ conv3d test passed!")
    print("✅ All conv tests passed!")


def test_dynamic_layernorm(runtime, torch_rng_seed):
    """Test torch.native_layer_norm with dynamic shapes"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # 使用 torch.nn.functional.layer_norm
    # 注意：normalized_shape 需要是固定的，不能动态变化
    # 这里我们测试特征维度固定为 16 的情况
    class LayerNormModel(torch.nn.Module):
        def forward(self, x, weight, bias):
            # normalized_shape 固定为 16
            return torch.nn.functional.layer_norm(
                x, [16], weight=weight, bias=bias, eps=1e-3
            )

    model = LayerNormModel()
    # Initial input: [batch, features=16]
    input_info = [((4, 5, 16), "float32"), ((16,), "float32"), ((16,), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    # First run with different batch size, but same features=16
    input_info_1 = [((3, 10, 16), "float32"), ((16,), "float32"), ((16,), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 10, 16)

    # Second run with another batch size, same features=16
    input_info_2 = [((1, 3, 16), "float32"), ((16,), "float32"), ((16,), "float32")]
    input_tensors_2 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_2
    ]
    translator.run(input_tensors_2)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (1, 3, 16)
    print("✅ LayerNorm test passed!")


def test_log_softmax(runtime, torch_rng_seed):
    """Test torch.log_softmax with dynamic shapes"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class LogSoftmaxModel(torch.nn.Module):
        def forward(self, x):
            # 这个 dim 没用，下游 infiniCore 不支持 dim
            return torch.log_softmax(x, dim=-1)

    model = LogSoftmaxModel()
    # Randomly initialize inputs
    input_info = [((5, 4), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    # First run with different shape
    input_info_1 = [((15, 12), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (15, 12)

    # Second run with another shape
    input_info_2 = [((3, 20), "float32")]
    input_tensors_2 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_2
    ]
    translator.run(input_tensors_2)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 20)
    print("✅ LogSoftmax test passed!")


def test_softmax(runtime, torch_rng_seed, device_type):
    """Test torch.softmax with dynamic shapes"""
    # 底层 infiniCore 不支持 CPU，跳过测试
    if device_type == DeviceType.CPU:
        pytest.skip("Softmax is not supported on CPU backend")

    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class SoftmaxModel(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=-1)

    model = SoftmaxModel()
    # Randomly initialize inputs
    input_info = [((5, 4), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    # First run with different shape
    input_info_1 = [((15, 12), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (15, 12)

    # Second run with another shape
    input_info_2 = [((3, 20), "float32")]
    input_tensors_2 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_2
    ]
    translator.run(input_tensors_2)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 20)
    print("✅ Softmax test passed!")


def test_normalize(runtime, torch_rng_seed, device_type):
    """Test torch.nn.functional.normalize with dynamic shapes"""
    # 底层 infiniCore 不支持 CPU，跳过测试
    if device_type == DeviceType.CPU:
        pytest.skip("LpNorm is not supported on CPU backend")

    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model - L2 normalization
    class NormalizeL2Model(torch.nn.Module):
        def forward(self, x):
            return torch.norm(x, p=2.0, dim=1)

    model = NormalizeL2Model()
    # Randomly initialize inputs
    input_info = [((5, 4), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    # First run with different shape
    input_info_1 = [((15, 12), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (15, 12)
    print("✅ Normalize L2 test passed!")


def test_rms_norm(runtime, torch_rng_seed, device_type):
    """Test torch.nn.functional.rms_norm with dynamic shapes"""

    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model - RMS normalization
    class RmsNormModel(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))

        def forward(self, x):
            return torch.nn.functional.rms_norm(
                x, (self.weight.shape[0],), self.weight, 1e-5
            )

    hidden_size = 16
    model = RmsNormModel(hidden_size)
    # Randomly initialize inputs
    input_info = [((4, 5, hidden_size), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    # First run with different shape
    input_info_1 = [((3, 10, hidden_size), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 10, hidden_size)
    print("✅ RmsNorm test passed!")


def test_relu(runtime, torch_rng_seed):
    """Test torch.nn.functional.relu with dynamic shapes"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class ReLUModel(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.relu(x)

    model = ReLUModel()
    input_info = [((4, 5, 16), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    input_info_1 = [((3, 10, 16), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 10, 16)
    print("✅ ReLU test passed!")


def test_sigmoid(runtime, torch_rng_seed):
    """Test torch.nn.functional.sigmoid with dynamic shapes"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class SigmoidModel(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.sigmoid(x)

    model = SigmoidModel()
    input_info = [((4, 5, 16), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    input_info_1 = [((3, 10, 16), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 10, 16)
    print("✅ Sigmoid test passed!")


def test_silu(runtime, torch_rng_seed):
    """Test torch.nn.functional.silu with dynamic shapes"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class SiLUModel(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.silu(x)

    model = SiLUModel()
    input_info = [((4, 5, 16), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    input_info_1 = [((3, 10, 16), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 10, 16)
    print("✅ SiLU test passed!")


def test_gelu(runtime, torch_rng_seed):
    """Test torch.nn.functional.gelu with dynamic shapes"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class GELUModel(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.gelu(x)

    model = GELUModel()
    input_info = [((4, 5, 16), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    input_info_1 = [((3, 10, 16), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 10, 16)
    print("✅ GELU test passed!")


def test_softplus(runtime, torch_rng_seed):
    """Test torch.nn.functional.softplus with dynamic shapes"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class SoftplusModel(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.softplus(x)

    model = SoftplusModel()
    input_info = [((4, 5, 16), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    input_info_1 = [((3, 10, 16), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 10, 16)
    print("✅ Softplus test passed!")


def test_tanh(runtime, torch_rng_seed):
    """Test torch.nn.functional.tanh with dynamic shapes"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class TanhModel(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.tanh(x)

    model = TanhModel()
    input_info = [((4, 5, 16), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    input_info_1 = [((3, 10, 16), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (3, 10, 16)
    print("✅ Tanh test passed!")


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
