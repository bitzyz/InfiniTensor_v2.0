import ctypes
import pyinfinitensor
from pyinfinitensor import GraphBuilder, Tensor, dtype_from_string
import torch
from torch import fx
from typing import Callable, Dict, List, Tuple, Optional, Union
from .converter import registry


class TorchFXTranslator:
    def __init__(self, runtime, custom_converters: Optional[Dict] = None):
        self.runtime = runtime
        self.builder = None 
        self.nodes_map: Dict[fx.Node, Any] = {} # 存储fx.Node映射关系，不论是Tensor还是Callable
        self.tensors: Dict[fx.Node, Tensor] = {} # 存储所有张量
        self.params: Dict[torch.Tensor, Tensor] = {} # 存储所有参数
        self.outputs: List[Tensor] = [] # 存储输出张量
        self.input_vars: Dict[str, Tensor] = {}
        self.named_modules = None
        self.shape_vars = {}  # 存储符号变量映射
        self.dynamic_dims = {}  # 存储动态维度信息
        if custom_converters:
            registry.update(custom_converters)


    def _process_call_module(self, node):
        module = self.named_modules[node.target]
        module_type = type(module)
        function = registry.get_module_converter(module_type)
        if function:
            try:
                self.nodes_map[node] = function
                function(self, node, module)
            except Exception as e:
                raise RuntimeError(
                    f"Converter for {module_type.__name__} failed: {str(e)}"
                )
        else:
            raise ValueError(f"Unsupported module type: {module_type.__name__}")
        

    def _create_input_tensors(self, input_list: List[torch.Tensor], dynamic_input_info: List[Tuple[Tuple, str]]) -> List:
        """创建输入张量"""
        # dynamic_input_info是通过从图文件中提取的动态形状信息，input_info是用户提供的静态形状信息
        # TODO:目前后端只支持静态形状信息，待支持动态形状信息后，需要重新设计这里的方案
        input_tensors = []
        if len(dynamic_input_info) != 0 and len(input_list) != len(dynamic_input_info):
            raise ValueError("Input info and dynamic input info should have the same length.")
        # 优先使用input_info
        if len(dynamic_input_info) != 0:
            for i, torch_tensor in enumerate(input_list):
                dtype = dtype_from_string(str(torch_tensor.dtype))
                tensor = self.builder.tensor(list(torch_tensor.size()), dtype)
                if torch_tensor.numel() > 0:
                    tensor.set_data(torch_tensor.data_ptr(), self.runtime)
                input_tensors.append(tensor)
                self.input_vars[f"inp_{i}"] = tensor
        else:
            for i, (shape, dtype) in enumerate(dynamic_input_info):
                tensor = self.builder.tensor(shape, dtype)
                input_tensors.append(tensor)
                self.input_vars[f"inp_{i}"] = tensor
        return input_tensors

    def _extract_fake_tensors(self, graph):
        """从FX图中提取FakeTensor信息"""
        fake_inputs = []
        
        for node in graph.nodes:
            if node.op != "placeholder":
                continue
            if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is not None:
                fake_tensor = node.meta["grapharg"].fake_tensor
                fake_inputs.append(fake_tensor)
            if "val" in node.meta and isinstance(node.meta["val"], torch._subclasses.fake_tensor.FakeTensor):
                fake_tensor = node.meta["val"]
                fake_inputs.append(fake_tensor)
            #TODO: 由于不同的torch compile方法会导致Node中meta信息不一致，可能需进一步调研，目前是借鉴了tvm现有实现
        
        return fake_inputs

    def _process_dynamic_shapes(self, fake_inputs):
        """处理动态形状"""
        dynamic_input_info: List[Tuple[Tuple, str]] = []

        for tensor in fake_inputs:
            shape = []
            for i, s in enumerate(tensor.shape):
                if hasattr(torch, 'SymInt') and isinstance(s, torch.SymInt) and not str(s).isdigit():
                    # 处理符号维度
                    sym_str = str(s)
                    if sym_str not in self.shape_vars:
                        # 创建符号变量（需要根据框架API调整）
                        self.shape_vars[sym_str] = f"symbolic_{sym_str}"
                    shape.append(self.shape_vars[sym_str])
                    
                    # 记录动态维度信息
                    self.dynamic_dims[sym_str] = {
                        'symbol': s,
                        'var': self.shape_vars[sym_str],
                        'dim_idx': i
                    }
                else:
                    # 具体维度
                    shape.append(int(s))
            
            dtype = dtype_from_string(str(tensor.dtype))
            dynamic_input_info.append((shape, dtype))
        
        return dynamic_input_info

    def _process_placeholder(self, node, input_tensors):
        """处理输入占位符节点"""
        if len(input_tensors) == 0:
            raise ValueError(
                f"Provided inputs is less than actual inputs. "
                f"Node {node.name} requires an input but no more inputs available."
            )
        tensor = input_tensors.pop(0)
        self.nodes_map[node] = tensor
        self.tensors[node] = tensor

    def _fetch_attr(self, model, target: str):
        """获取模型属性"""
        target_atoms = target.split('.')
        attr_itr = model
        
        try:
            for atom in target_atoms:
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced non-existing target: {target}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr
        except Exception as e:
            raise RuntimeError(f"Failed to fetch attribute {target}: {e}")

    def _process_get_attr(self, node, fx_module):
        """处理属性获取节点"""
        attr_value = self._fetch_attr(fx_module, node.target)
        
        if isinstance(attr_value, torch.Tensor):
            # 如果是参数或缓冲区张量
            if attr_value not in self.params:
                self.params[attr_value] = self.builder.tensor(attr_value.shape, dtype_from_string(attr_value.dtype))
                self.params[attr_value].set_data(attr_value.data_ptr(), self.runtime)
                self.nodes_map[node] = self.params[attr_value]
                self.tensors[node] = self.params[attr_value]
        else:
            raise ValueError(f"Unsupported attribute type: {type(attr_value)}")
    
    def _process_call_function(self, node):
        """处理函数调用节点"""
        func_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
        function = registry.get_method_converter(func_name)
        if function:
            try:
                self.nodes_map[node] = function
                function(self, node)
            except Exception as e:
                raise RuntimeError(
                    f"Converter for {func_name} failed: {str(e)}"
                )
        else:
            raise ValueError(f"Unsupported function: {func_name}")

    def _process_call_method(self, node):
        """处理方法调用节点"""
        method_name = node.target
        function = registry.get_method_converter(method_name)
        if function:
            try:
                self.nodes_map[node] = function
                function(self, node)
            except Exception as e:
                raise RuntimeError(
                    f"Converter for {method_name} failed: {str(e)}"
                )
        else:
            raise ValueError(f"Unsupported method: {method_name}")

    def _retrieve_args(self, node):
        if isinstance(node, fx.Node):
            return node
        elif isinstance(node, list):
            return [self._retrieve_args(n) for n in node]
        elif isinstance(node, tuple):
            return tuple(self._retrieve_args(n) for n in node)
        elif isinstance(node, dict):
            return {self._retrieve_args(k): self._retrieve_args(v) for k, v in node.items()}
        elif node is None:
            return None
        else:
            return node

    def _process_output(self, node):
        """处理输出节点"""
        args = self._retrieve_args(node.args)
        assert len(args) == 1
        if isinstance(args[0], (tuple, list)):
            for arg in args[0]:
                self.outputs.append(self.tensors[arg])
        else:
            self.outputs.append(self.tensors[args[0]])
        
    def _tensor_from_torch_info(self, torch_info):
        """从Torch信息创建张量"""
        data_ptr_int, shape, stride, dtype_str, storage_size = torch_info
        dtype = getattr(torch, dtype_str)
        buf_type = ctypes.c_char * storage_size
        buf = buf_type.from_address(data_ptr_int)
        t = torch.frombuffer(buf, dtype=dtype)
        t = t.as_strided(size=shape, stride=stride)
        return t

    def import_from_fx(self, fx_module, input_list: List[torch.Tensor]):
        """
        导入FX图到计算图框架
        
        Args:
            fx_module: PyTorch FX图模块
            input_list: 输入张量列表
        """

        self.builder = GraphBuilder(self.runtime)
        self.named_modules = dict(fx_module.named_modules())
        
        graph = fx_module.graph
        # 提取符号形状信息
        fake_inputs = self._extract_fake_tensors(graph)
        dynamic_input_info = self._process_dynamic_shapes(fake_inputs)
        # 创建输入张量
        inputs = self._create_input_tensors(input_list, dynamic_input_info)
        # 创建params
        for _, param in fx_module.named_parameters():
            if isinstance(param, torch.Tensor):
                self.params[param] = self.builder.tensor(param.shape, dtype_from_string(str(param.dtype)))
                self.params[param].set_data(param.data_ptr(), self.runtime)
        
        # 处理FX图节点
        for node in graph.nodes:
            if node.op == "placeholder":
                self._process_placeholder(node, inputs)
            elif node.op == "call_function":
                self._process_call_function(node)
            elif node.op == "call_module": 
                self._process_call_module(node)
            elif node.op == "call_method":
                self._process_call_method(node)
            elif node.op == "get_attr":
                self._process_get_attr(node, fx_module)
            elif node.op == "output":
                self._process_output(node)
                break
            else:
                raise ValueError(f"Unsupported node op: {node.op}")
        print(self.builder.to_string())
        
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
