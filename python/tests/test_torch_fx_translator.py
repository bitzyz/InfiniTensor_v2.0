import torch
import torch.fx
import torch.nn as nn
import torch._dynamo as dynamo
import infinitensor
from infinitensor import TorchFXTranslator

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(4, 5)
        self.linear_2 = torch.nn.Linear(5, 3, bias=False)

    def forward(self, x):
        return self.linear_2(self.linear_1(x))

m = MyModule()
dy_gm = dynamo.export(m, torch.randn(5, 4)).graph_module
runtime = Runtime()
translator = TorchFXTranslator(runtime)
input_info = [((5,4), "float32")]
translator.import_from_fx(dy_gm, input_info)
runtime.run()
outputs = translator.get_output()
print(outputs)