import pytest
import torch
import torch.nn as nn
import numpy as np
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType


def test_basic_matmul(runtime, torch_rng_seed):
    """直接使用conftest.py中定义的fixtures"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")
    
    # 创建简单模型
    class MatmulModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)
    
    model = MatmulModel()
    # 随机初始化输入,传入形状可以与真实传入值不一样，但是数据类型需要一致
    input_info = [((5, 4), "float32"), ((4, 5), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]
    
    # 创建转换器
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)
    # 运行
    translator.run(input_tensors)
    
    # 获取输出
    outputs = translator.get_outputs()
    
    # 验证
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 5, 3)
    print("✅ Test passed!")

def test_dynamic_matmul(runtime, torch_rng_seed):
    """直接使用conftest.py中定义的fixtures"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")
    
    # 创建简单模型
    class MatmulModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)
    
    model = MatmulModel()
    # 随机初始化输入,传入形状可以与真实传入值不一样，但是数据类型需要一致
    input_info = [((5, 4), "float32"), ((4, 7), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]
    
    # 创建转换器
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


if __name__ == "__main__":
    # 可以直接运行这个文件
    import sys
    
    # 使用pytest运行所有测试
    exit_code = pytest.main([
        __file__,
        "-v",  # 详细输出
        "-s",  # 显示print输出
        "--tb=short",  # 简化的错误回溯
    ])
    
    sys.exit(0 if exit_code == 0 else 1)
