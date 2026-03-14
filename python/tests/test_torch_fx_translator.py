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
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class ClipModel(torch.nn.Module):
        def forward(self, x, min_val, max_val):
            return torch.clip(x, min=min_val, max=max_val)

    model = ClipModel()

    # Randomly initialize inputs, passed shapes can differ from actual values, but data types must match
    input_info = [((5, 4), "float32"), ((5, 4), "float32"), ((5, 4), "float32")]
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
    assert outputs[0].shape == (5, 4)
    print("✅ Test passed!")


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
