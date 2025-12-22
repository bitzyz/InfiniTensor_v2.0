import torch.nn as nn
from .registry import registry

#https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml

@registry.register("matmul","default")
def convert_matmul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.gemm(a, b, None)
