import pytest
import torch
import torch.nn as nn
import torch._dynamo as dynamo
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
    x = torch.randn(5, 4)
    y = torch.randn(4, 3)
    
    # 导出FX图
    graph_module = dynamo.export(model, x, y).graph_module
    
    # 创建转换器
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(graph_module, [x, y])
    
    # 运行
    runtime.run(translator.builder.graph)
    
    # 获取输出
    outputs = translator.get_outputs()
    
    # 验证
    assert len(outputs) == 1
    assert outputs[0].shape == (5, 3)
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
