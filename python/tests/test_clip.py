import pytest
import torch
import torch.nn as nn
import numpy as np
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType


def test_clip(runtime, torch_rng_seed):
    """Test the Clip operator integration."""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Construct a simple graph with Clip operator
    class ClipModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.clamp(x, min=-1.0, max=1.0)

    model = ClipModule()

    # Create input tensor
    input_shape = (5, 4)
    input_tensor = torch.randn(input_shape, dtype=torch.float32)

    # Expected output from PyTorch
    expected_output = model(input_tensor).numpy()

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, [input_tensor])

    # Run
    translator.run([input_tensor])

    # Get outputs
    outputs = translator.get_outputs()

    # Verify
    assert len(outputs) == 1
    actual_output = outputs[0].numpy()
    assert actual_output.shape == expected_output.shape

    # Use np.allclose for element-wise comparison with a tolerance
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-4)
    print("✅ Clip operator test passed!")


if __name__ == "__main__":
    import sys

    exit_code = pytest.main([__file__, "-v", "-s"])
    sys.exit(0 if exit_code == 0 else 1)
