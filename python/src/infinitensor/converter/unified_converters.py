import torch.nn as nn
from .registry import registry

# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml


@registry.register("matmul", "default")
def convert_matmul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.gemm(a, b, None)


@registry.register("add", "Tensor")
def convert_add(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.add(a, b, None)


@registry.register("mul", "Tensor")
def convert_mul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.mul(a, b, None)


@registry.register("sub", "Tensor")
def convert_sub(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.sub(a, b, None)


@registry.register("clamp", "Tensor")
def convert_clamp(translator, node):
    t = translator.tensors[node.args[0]]
    mmin = translator.tensors[node.args[1]]
    mmax = translator.tensors[node.args[2]]
    translator.tensors[node] = translator.builder.clamp(t, mmin, mmax, None)


@registry.register("conv1d", "default")
def convert_conv1d(translator, node):
    x = translator.tensors[node.args[0]]
    weight = translator.tensors[node.args[1]]
    b = (
        None
        if node.args[2] not in translator.tensors
        else translator.tensors[node.args[2]]
    )

    stride = [
        1,
    ]
    padding = [
        0,
    ]
    dilation = [
        1,
    ]

    if len(node.args) >= 4:
        stride = [
            int(node.args[3][0]),
        ]
    if len(node.args) >= 5:
        padding = [
            int(node.args[4][0]),
        ]
    if len(node.args) >= 6:
        dilation = [
            int(node.args[5][0]),
        ]
    translator.tensors[node] = translator.builder.conv(
        x, weight, b, stride, padding, dilation, 1, None
    )


@registry.register("conv2d", "default")
def convert_conv1d(translator, node):
    x = translator.tensors[node.args[0]]
    weight = translator.tensors[node.args[1]]
    b = (
        None
        if node.args[2] not in translator.tensors
        else translator.tensors[node.args[2]]
    )

    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]

    if len(node.args) >= 4:
        stride = [int(node.args[3][0]), int(node.args[3][1])]
    if len(node.args) >= 5:
        padding = [int(node.args[4][0]), int(node.args[4][1])]
    if len(node.args) >= 6:
        dilation = [int(node.args[5][0]), int(node.args[5][1])]
    translator.tensors[node] = translator.builder.conv(
        x, weight, b, stride, padding, dilation, 2, None
    )


@registry.register("conv3d", "default")
def convert_conv1d(translator, node):
    x = translator.tensors[node.args[0]]
    weight = translator.tensors[node.args[1]]
    b = (
        None
        if node.args[2] not in translator.tensors
        else translator.tensors[node.args[2]]
    )

    stride = [1, 1, 1]
    padding = [0, 0, 0]
    dilation = [1, 1, 1]

    if len(node.args) >= 4:
        stride = [int(node.args[3][0]), int(node.args[3][1]), int(node.args[3][1])]
    if len(node.args) >= 5:
        padding = [int(node.args[4][0]), int(node.args[4][1]), int(node.args[4][2])]
    if len(node.args) >= 6:
        dilation = [int(node.args[5][0]), int(node.args[5][1]), int(node.args[4][2])]
    translator.tensors[node] = translator.builder.conv(
        x, weight, b, stride, padding, dilation, 3, None
    )


@registry.register("layer_norm", "default")
def convert_layer_norm(translator, node):
    x = translator.tensors[node.args[0]]
    # shape下游infiniCore不支持
    shape = node.args[1]
    # 下游要求weight必须存在，bias不一定
    weight = translator.tensors[node.args[2]]
    bias = (
        None
        if node.args[3] not in translator.tensors
        else translator.tensors[node.args[3]]
    )
    eps = 1e-5
    if len(node.args) >= 5:
        eps = node.args[4]
    translator.tensors[node] = translator.builder.layer_norm(x, weight, bias, eps)


@registry.register("log_softmax", "int")
def convert_log_softmax(translator, node):
    x = translator.tensors[node.args[0]]
    dim = node.args[1]
    translator.tensors[node] = translator.builder.log_softmax(x, dim)


@registry.register("softmax", "int")
def convert_softmax(translator, node):
    x = translator.tensors[node.args[0]]
    axis = node.args[1]
    translator.tensors[node] = translator.builder.softmax(x, axis)


@registry.register("linalg_vector_norm", "default")
def convert_normalize(translator, node):
    x = translator.tensors[node.args[0]]
    p = 2.0
    dim = 1
    if len(node.args) >= 2:
        p = node.args[1]
    if len(node.args) >= 3:
        dim = node.args[2][0]
    # if len(node.args) >= 4:
    #     eps = node.args[3]
    translator.tensors[node] = translator.builder.lp_norm(x, dim, int(p))


@registry.register("rms_norm", "default")
def convert_rms_norm(translator, node):
    x = translator.tensors[node.args[0]]
    normalized_shape = node.args[1]
    weight = translator.tensors[node.args[2]]
    eps = 1e-5
    if len(node.args) >= 4:
        eps = node.args[3]
    translator.tensors[node] = translator.builder.rms_norm(x, weight, eps)


# Unary operations
@registry.register("relu", "default")
def convert_relu(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.relu(x)


@registry.register("sigmoid", "default")
def convert_sigmoid(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.sigmoid(x)


@registry.register("silu", "default")
def convert_silu(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.silu(x)


@registry.register("gelu", "default")
def convert_gelu(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.gelu(x)


@registry.register("softplus", "default")
def convert_softplus(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.softplus(x)


@registry.register("tanh", "default")
def convert_tanh(translator, node):
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.tanh(x)
