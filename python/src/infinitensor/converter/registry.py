import inspect
from typing import Dict, Callable, Optional
import torch.nn as nn
import torch
from torch import fx


class ConverterRegistry:

    def __init__(self):
        self._module_converters: Dict[torch.nn.Module, Callable[[fx.Node], None]] = {}
        self._method_converters: Dict[str, Callable[[fx.Node], None]] = {}

    def register_module(self, module_class):
        """装饰器：注册模块转换器"""

        def decorator(func):
            self._module_converters[module_class] = func
            return func

        return decorator

    def register_method(self, method_name: str):
        """装饰器：注册方法和函数转换器"""

        def decorator(func):
            self._method_converters[method_name] = func
            return func

        return decorator

    def get_module_converter(self, module_class) -> Optional[Callable]:
        """获取模块转换器"""
        # 也检查父类
        if module_class in self._module_converters:
            return self._module_converters[module_class]
        return None

    def get_method_converter(self, method_name: str) -> Optional[Callable]:
        """获取方法和函数转换器"""
        if method_name in self._method_converters:
            return self._method_converters[method_name]
        return None

    def update(self, custom_converters: Dict):
        """更新转换器"""
        for key, converter in custom_converters.items():
            if inspect.isclass(key) and issubclass(key, nn.Module):
                self._module_converters[key] = converter
            elif isinstance(key, str):
                self._method_converters[key] = converter

    def clear(self):
        """清空所有转换器"""
        self._module_converters.clear()
        self._method_converters.clear()

    def list_all_converters(self):
        """列出所有转换器"""
        return {
            "modules": list(self._module_converters.keys()),
            "methods": list(self._method_converters.keys()),
        }


# 全局注册器实例
registry = ConverterRegistry()
