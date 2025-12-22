import ctypes
import pyinfinitensor
from pyinfinitensor import GraphBuilder, Tensor, dtype_from_string, Runtime, ShapeExpr, StrideExpr
import torch
from torch import fx
from torch.export import export, Dim
from typing import Callable, Dict, List, Tuple, Optional, Union
from .converter import registry
import inspect


class TorchFXTranslator:
    def __init__(self, runtime: Runtime, custom_converters: Optional[Dict] = None):
        self.runtime = runtime
        self.module = None
        self.builder = None
        self.nodes_map: Dict[fx.Node, Any] = (
            {}
        )  # 存储fx.Node映射关系，不论是Tensor还是Callable
        self.tensors: Dict[fx.Node, Tensor] = {}  # 存储所有张量
        self.params: Dict[torch.Tensor, Tensor] = {}  # 存储所有参数
        self.outputs: List[Tensor] = []  # 存储输出张量
        self.input_vars: Dict[str, Tensor] = {}
        self.symbols = {}  # 符号 -> {'var': 变量名, 'value': 具体值, 'info': 详细信息}
        self.dynamic_input_infos: List[Tuple[Tuple, Tuple, str]] = []  # 动态输入信息(shape, stride, dtype)
        if custom_converters:
            registry.update(custom_converters)

    def _add_symbol(self, symbol_str, input_idx, dim_idx):
        """添加符号信息"""
        if symbol_str in self.symbols:
            self.symbols[symbol_str]["info"]["input_idx"].append(input_idx)
            self.symbols[symbol_str]["info"]["dim_idx"].append(dim_idx)
        else:
            var_name = f"symbolic_{symbol_str}"
            self.symbols[symbol_str] = {
                "var": var_name,
                "value": None,  # 初始化为None，表示未绑定
                "info": {
                    "input_idx": [input_idx],
                    "dim_idx": [dim_idx],
                },
            }

    def _clear_symbols(self):
        """清空符号信息"""
        for symbol_str in self.symbols:
            self.symbols[symbol_str]["value"] = None

    def _add_dynamic_shapes(self, model, input_list):
        """
        为每个 Tensor 生成:
        arg_{i}: {0: Dim.AUTO, 1: Dim.AUTO, ...}
        """
        sig = inspect.signature(model.forward)
        param_names = [p.name for p in sig.parameters.values() if p.name != "self"]
        assert(len(param_names) == len(input_list))
        dynamic_shapes = {}
        for idx, (p, t) in enumerate(zip(param_names, input_list)):
            if not isinstance(t, torch.Tensor):
                raise ValueError("input is not torch Tensor")
            dynamic_shapes[p] = {
                dim: Dim.AUTO for dim in range(t.dim())
            }
        return dynamic_shapes

    def _create_input_tensors(
        self, input_list: List[torch.Tensor], is_real_tensor: bool
    ) -> List:
        """创建输入张量"""
        # dynamic_input_infos是通过从图文件中提取的动态形状信息，input_info是用户提供的静态形状信息
        input_tensors = []
        if len(self.dynamic_input_infos) != 0 and len(input_list) != len(
            self.dynamic_input_infos
        ):
            raise ValueError(
                "Input info and dynamic input info should have the same length."
            )
        if is_real_tensor:
            for i, torch_tensor in enumerate(input_list):
                dtype = dtype_from_string(str(torch_tensor.dtype))
                tensor = self.builder.tensor(
                    ShapeExpr(list(torch_tensor.size())), dtype
                )
                if torch_tensor.numel() > 0:
                    tensor.set_data(torch_tensor.data_ptr(), self.runtime)
                input_tensors.append(tensor)
                self.input_vars[f"inp_{i}"] = tensor
        else:
            for i, (shape, stride, dtype) in enumerate(self.dynamic_input_infos):
                tensor = self.builder.tensor(ShapeExpr(shape), dtype, StrideExpr(stride))
                input_tensors.append(tensor)
                self.input_vars[f"inp_{i}"] = tensor
        return input_tensors

    def _process_dynamic_shapes(self, fake_inputs):
        """处理动态形状"""
        for i, tensor in enumerate(fake_inputs.values()):
            shape = tensor.shape
            stride = tensor.stride
            assert(len(shape) == len(stride))
            tensor_shape = []
            tensor_stride = []
            dtype = dtype_from_string(str(tensor.dtype))
            for j, (dim, st) in enumerate(zip(shape, stride[::-1])):
                # 处理shape信息
                if (
                    hasattr(torch, "SymInt")
                    and isinstance(dim, torch.SymInt)
                    and not str(dim).isdigit()
                ):
                    # 处理符号维度
                    sym_str = str(dim)
                    self._add_symbol(sym_str, i, j)
                    tensor_shape.append(self.symbols[sym_str]["var"])
                else:
                    # 具体维度
                    tensor_shape.append(int(dim))
                # 处理stride信息
                if (
                    hasattr(torch, "SymInt")
                    and isinstance(st, torch.SymInt)
                    and not str(st).isdigit()
                ):
                    # 处理符号维度
                    sym_str = str(st)
                    assert(self.symbols.get(sym_str))
                    tensor_stride.insert(0, self.symbols[sym_str]["var"])
                else:
                    # 具体维度
                    tensor_stride.insert(0, int(st))

            self.dynamic_input_infos.append((tensor_shape, tensor_stride, dtype))

    def _process_call_function(self, node):
        """处理函数调用节点"""
        target = node.target
        if hasattr(target,"_overloadpacket"):
            op_name = str(target._overloadpacket).split('.')[-1]
            overload = target._overloadname
            function = registry.get_method_converter(op_name, overload)
        else:
            if hasattr(target, "__name__"):
                op_base_name = target.__name__
            else:
                op_base_name = str(target)
            function = registry.get_method_converter(op_base_name)
        if function:
            try:
                self.nodes_map[node] = function
                function(self, node)
            except Exception as e:
                raise RuntimeError(f"Converter for {func_name} failed: {str(e)}")
        else:
            raise ValueError(f"Unsupported function: {func_name}")

    def _process_output(self, node):
        """处理输出节点"""
        args = self._retrieve_args(node.args)
        assert len(args) == 1
        if isinstance(args[0], (tuple, list)):
            for arg in args[0]:
                self.outputs.append(self.tensors[arg])
        else:
            self.outputs.append(self.tensors[args[0]])

    def _retrieve_args(self, node):
        if isinstance(node, fx.Node):
            return node
        elif isinstance(node, list):
            return [self._retrieve_args(n) for n in node]
        elif isinstance(node, tuple):
            return tuple(self._retrieve_args(n) for n in node)
        elif isinstance(node, dict):
            return {
                self._retrieve_args(k): self._retrieve_args(v) for k, v in node.items()
            }
        elif node is None:
            return None
        else:
            return node

    def _tensor_from_torch_info(self, torch_info):
        """从Torch信息创建张量"""
        data_ptr_int, shape, stride, dtype_str, storage_size = torch_info
        dtype = getattr(torch, dtype_str)
        buf_type = ctypes.c_char * storage_size
        buf = buf_type.from_address(data_ptr_int)
        t = torch.frombuffer(buf, dtype=dtype)
        t = t.as_strided(size=shape, stride=stride)
        return t

    def _extract_graph_signature(self):
        fake_inputs = {}
        def transform_parameter_string(s):
            return re.sub(r'_', '.', re.sub(r'^p_', '', s))
        def transform_buffer_string(s):
            return re.sub(r'_', '.', re.sub(r'^b_', '', s))
        nodes = list(self.module.graph_module.graph.nodes)
        for i, spec in enumerate(self.module.graph_signature.input_specs):
            kind = spec.kind.name
            node = nodes[i]
            name = spec.arg.name
            if kind == "PARAMETER":
                assert(node.op == "placeholder" and isinstance(
                node.meta["val"], torch._subclasses.fake_tensor.FakeTensor
            ))
                shape_expr = ShapeExpr(node.meta["tensor_meta"].shape)
                stride_expr = StrideExpr(node.meta["tensor_meta"].stride)
                dtype = dtype_from_string(str(node.meta["tensor_meta"].dtype))
                self.params[name] = self.builder.tensor(shape_expr, dtype, stride_expr)
                self.params[name].set_data(self.module.state_dict[transform_parameter_string(name)].data_ptr(),self.runtime)
                self.nodes_map[node] = self.params[name]
                self.tensors[node] = self.params[name]
            elif kind == "BUFFER":
                if len(node.users) == 0:
                    continue
                assert(node.op == "placeholder" and isinstance(
                node.meta["val"], torch._subclasses.fake_tensor.FakeTensor
            ))
                shape_expr = ShapeExpr(node.meta["tensor_meta"].shape)
                stride_expr = StrideExpr(node.meta["tensor_meta"].stride)
                dtype = dtype_from_string(str(node.meta["tensor_meta"].dtype))
                self.params[name] = self.builder.tensor(shape_expr, dtype, stride_expr)
                self.params[name].set_data(self.module.state_dict[transform_buffer_string(name)].data_ptr(),self.runtime)
                self.nodes_map[node] = self.params[name]
                self.tensors[node] = self.params[name]
            elif kind == "USER_INPUT":
                if "val" in node.meta and isinstance(
                node.meta["val"], torch._subclasses.fake_tensor.FakeTensor):
                    fake_tensor = node.meta["tensor_meta"]
                    fake_inputs[node] = fake_tensor
            else:
                raise ValueError(f"Unsupported input kind: {kind}")
        
        return fake_inputs


    def import_from_fx(
        self, model, input_list: List[torch.Tensor], is_real_tensor: bool = False
    ):
        """
        导入FX图到计算图框架

        Args:
            model: PyTorch Model
            input_list: 输入张量列表
        """

        self.builder = GraphBuilder(self.runtime)
        dynamic_shapes = self._add_dynamic_shapes(model, input_list)
        try:
            self.module = export(model, tuple(input_list), dynamic_shapes=dynamic_shapes)
        except:
            raise RuntimeError("Failed to export the PyTorch model to FX.")

        # 解析graph_signature，提取params、buffers、inputs、outputs
        fake_inputs = self._extract_graph_signature()

        # 提取符号形状信息
        self._process_dynamic_shapes(fake_inputs)
        # 创建输入张量
        inputs = self._create_input_tensors(input_list, is_real_tensor)
        for (node, tensor) in zip(fake_inputs.keys(), inputs):
            self.nodes_map[node] = tensor
            self.tensors[node] = tensor

        # 处理FX图节点
        for node in self.module.graph_module.graph.nodes:
            if node.op == "placeholder":
                continue
            elif node.op == "call_function":
                self._process_call_function(node)
            elif node.op == "output":
                self._process_output(node)
                break
            else:
                raise ValueError(f"Unsupported node op: {node.op}")

        # print(self.builder.to_string())

    def run(self, input_list: List[torch.Tensor]):
        """
        运行计算图

        Args:
            input_list: 输入张量列表
        """
        self._clear_symbols()
        if len(input_list) != len(self.dynamic_input_infos):
            raise ValueError("The input tensor len is not equal the model input len")
        for i, tensor in enumerate(input_list):
            if len(tensor.shape) != len(self.dynamic_input_infos[i][0]):
                raise ValueError(
                    f"The input tensor shape len is not equal the model input shape len, input {i}"
                )
            shape = []
            for j, s in enumerate(tensor.shape):
                shape_ele = self.dynamic_input_infos[i][0][j]
                if isinstance(shape_ele, str):
                    shape_ele = shape_ele.replace("symbolic_", "", 1)
                    if self.symbols[shape_ele]["value"] is None:
                        self.symbols[shape_ele]["value"] = s
                    else:
                        if self.symbols[shape_ele]["value"] != s:
                            raise ValueError(
                                f"The input {i}, dim {j} shape should equal {s}, but is {self.symbols[shape_ele]['value']}"
                            )
                else:
                    if s != shape_ele:
                        raise ValueError(
                            f"The input {i}, dim {j} shape should equal {shape_ele}, but is {s}"
                        )
                shape.append(s)
            self.input_vars[f"inp_{i}"].set_shape(shape)
            self.input_vars[f"inp_{i}"].set_data(tensor.data_ptr(), self.runtime)
        self.runtime.run(self.builder.graph)

    def get_outputs(self) -> List[torch.Tensor]:
        """
        获取输出Torch张量

        Returns:
            outputs: 输出Torch张量列表
        """
        outputs = []
        for output in self.outputs:
            torch_info = output.to_torch_info(self.runtime)
            torch_tensor = self._tensor_from_torch_info(torch_info)
            outputs.append(torch_tensor)
        return outputs
