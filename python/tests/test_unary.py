import pytest
import torch
import torch.nn as nn
import numpy as np
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType


@pytest.mark.parametrize(
    "op_name, torch_op",
    [
        ("relu", torch.relu),
        ("sigmoid", torch.sigmoid),
        ("tanh", torch.tanh),
        ("gelu", torch.nn.functional.gelu),
        ("silu", torch.nn.functional.silu),
        ("softplus", torch.nn.functional.softplus),
    ],
)
def test_unary(runtime, torch_rng_seed, op_name, torch_op):
    print(f"Testing {op_name} with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class UnaryModule(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x):
            return self.op(x)

    model = UnaryModule(torch_op)

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
    print(f"✅ {op_name} operator test passed!")


if __name__ == "__main__":
    import sys

    exit_code = pytest.main([__file__, "-v", "-s"])
    sys.exit(0 if exit_code == 0 else 1)
