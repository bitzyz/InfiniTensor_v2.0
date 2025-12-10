import torch
import torch.fx
import torch.nn as nn
import torch._dynamo as dynamo
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

m = MyModule()
dy_gm = dynamo.export(m, torch.randn(5, 4), torch.randn(4, 3)).graph_module
# print(dy_gm.graph.print_tabular())
# runtime = Runtime.get_instance()
# Runtime.init()
# runtime.init_thread_context(DeviceType.CUDA)
runtime = Runtime.setup(DeviceType.CUDA)
translator = TorchFXTranslator(runtime)
a = torch.randn(5, 4)
b = torch.randn(4, 3)
input_list = [a, b]
translator.import_from_fx(dy_gm, input_list)
runtime.run(translator.builder.graph)
# outputs = translator.get_outputs()
# print(outputs[0])
# torch_outputs = m(a, b)
# print(torch_outputs)
# assert torch.allclose(outputs[0], torch_outputs)