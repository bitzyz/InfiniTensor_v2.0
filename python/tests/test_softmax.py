import pytest
import torch
import torch.nn as nn
import numpy as np
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax(runtime, torch_rng_seed, axis):
    print(f"Testing softmax axis={axis} with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class SoftmaxModule(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.softmax = nn.Softmax(dim=dim)

        def forward(self, x):
            return self.softmax(x)

    model = SoftmaxModule(axis)

    input_shape = (2, 4, 8)
    input_tensor = torch.randn(input_shape, dtype=torch.float32)

    expected_output = model(input_tensor).detach().numpy()

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(
        model, [input_tensor], is_real_tensor=True, dynamic_shapes=False
    )
    translator.run([input_tensor])
    outputs = translator.get_outputs()

    assert len(outputs) == 1
    actual_output = outputs[0].numpy()
    assert actual_output.shape == expected_output.shape

    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-4)
    print(f"✅ Softmax axis={axis} test passed!")


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_log_softmax(runtime, torch_rng_seed, axis):
    print(f"Testing log_softmax axis={axis} with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class LogSoftmaxModule(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.log_softmax = nn.LogSoftmax(dim=dim)

        def forward(self, x):
            return self.log_softmax(x)

    model = LogSoftmaxModule(axis)

    input_shape = (2, 4, 8)
    input_tensor = torch.randn(input_shape, dtype=torch.float32)

    expected_output = model(input_tensor).detach().numpy()

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(
        model, [input_tensor], is_real_tensor=True, dynamic_shapes=False
    )
    translator.run([input_tensor])
    outputs = translator.get_outputs()

    assert len(outputs) == 1
    actual_output = outputs[0].numpy()
    assert actual_output.shape == expected_output.shape

    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-4)
    print(f"✅ LogSoftmax axis={axis} test passed!")


if __name__ == "__main__":
    import sys

    exit_code = pytest.main([__file__, "-v", "-s"])
    sys.exit(0 if exit_code == 0 else 1)
