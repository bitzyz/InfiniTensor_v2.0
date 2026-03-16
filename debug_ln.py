import torch
import torch.nn as nn
from torch.export import export


class LayerNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm([16, 32], elementwise_affine=True)

    def forward(self, x):
        return self.ln(x)


model = LayerNormModule()
input_tensor = torch.randn((2, 8, 16, 32), dtype=torch.float32)
ep = export(model, (input_tensor,))
for node in ep.graph.nodes:
    if node.op == "call_function":
        print(node.target, node.args, node.kwargs)
