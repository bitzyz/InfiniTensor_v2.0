import pytest
import torch
import torch.nn as nn
import numpy as np
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType


@pytest.mark.parametrize("p", [1.0, 2.0, float("inf")])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_lpnorm(runtime, torch_rng_seed, p, dim, keepdim):
    print(
        f"Testing LpNorm p={p}, dim={dim}, keepdim={keepdim} with runtime on device: {runtime}"
    )
    print(f"Random seed: {torch_rng_seed}")

    class LpNormModule(torch.nn.Module):
        def __init__(self, p, dim, keepdim):
            super().__init__()
            self.p = p
            self.dim = dim
            self.keepdim = keepdim

        def forward(self, x):
            return torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)

    model = LpNormModule(p, dim, keepdim)

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
    print(f"✅ LpNorm p={p}, dim={dim}, keepdim={keepdim} test passed!")


if __name__ == "__main__":
    import sys

    exit_code = pytest.main([__file__, "-v", "-s"])
    sys.exit(0 if exit_code == 0 else 1)
