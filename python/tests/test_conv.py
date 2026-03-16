import pytest
import torch
import torch.nn as nn
import numpy as np
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType


def test_conv(runtime, torch_rng_seed):
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class ConvModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 2d conv: in_channels=3, out_channels=16, kernel_size=3, padding=1
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)

        def forward(self, x):
            return self.conv(x)

    model = ConvModule()

    input_shape = (1, 3, 32, 32)
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
    print("✅ Conv operator test passed!")


if __name__ == "__main__":
    import sys

    exit_code = pytest.main([__file__, "-v", "-s"])
    sys.exit(0 if exit_code == 0 else 1)
