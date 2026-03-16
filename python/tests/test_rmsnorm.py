import pytest
import torch
import torch.nn as nn
import numpy as np
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType, GraphBuilder, Tensor


def test_rmsnorm(runtime, torch_rng_seed):
    print(f"Testing RMSNorm with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    hidden_size = 32
    eps = 1e-6

    input_shape = (2, 8, hidden_size)
    input_tensor = torch.randn(input_shape, dtype=torch.float32)
    weight_tensor = torch.ones((hidden_size,), dtype=torch.float32)

    # Reference implementation
    input_dtype = input_tensor.dtype
    hidden_states = input_tensor.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    expected_output = (weight_tensor * hidden_states).to(input_dtype).numpy()

    # Build Graph manually
    builder = GraphBuilder(runtime)

    # Create input tensors
    # We need to set data for them
    input_t = builder.tensor(
        infinitensor.ShapeExpr(list(input_shape)),
        infinitensor.dtype_from_string("float32"),
    )
    weight_t = builder.tensor(
        infinitensor.ShapeExpr([hidden_size]), infinitensor.dtype_from_string("float32")
    )

    input_t.set_data(input_tensor.data_ptr(), runtime)
    weight_t.set_data(weight_tensor.data_ptr(), runtime)

    # Add RMSNorm op
    output_t = builder.rms_norm(input_t, weight_t, eps)

    # Run
    runtime.run(builder.graph)

    # Verify
    ptr, shape, stride, dtype_str, size = output_t.to_torch_info(runtime)
    # Create torch tensor from ptr?
    # This might be unsafe if we don't manage lifetime.
    # But for test it's fine.
    # Actually, we can use `ctypes` to copy from ptr to numpy.
    import ctypes

    buffer = (ctypes.c_float * (size // 4)).from_address(ptr)
    actual_output_np = np.ctypeslib.as_array(buffer).reshape(shape)

    np.testing.assert_allclose(actual_output_np, expected_output, rtol=1e-5, atol=1e-4)
    print("✅ RMSNorm operator test passed!")


if __name__ == "__main__":
    import sys

    exit_code = pytest.main([__file__, "-v", "-s"])
    sys.exit(0 if exit_code == 0 else 1)
