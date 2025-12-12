import torch.nn as nn
from .registry import registry


@registry.register_module(nn.Linear)
def convert_linear(translator, node, module):
    x = translator.tensors[node.args[0]]
    module = translator.named_modules[node.target]
    weight = translator.params[module.weight]
    bias = translator.params.get(module.bias, None)
    translator.tensors[node] = translator.builder.gemm(x, weight, bias, transB=True)


@registry.register_method("matmul")
def convert_matmul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.gemm(a, b, None)
