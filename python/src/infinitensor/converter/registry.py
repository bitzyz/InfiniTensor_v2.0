import inspect
from typing import Dict, Callable, Optional
import torch.nn as nn
import torch
from torch import fx


class ConverterRegistry:

    def __init__(self):
        # { "aten.matmul": { "default": fn, "out": fn, None: fn } }
        self._method_converters: Dict[str, Dict[Optional[str], Callable]] = {}

    def register(self, op_name: str, overload: Optional[str] = None):
        """装饰器：注册方法和函数转换器"""

        def decorator(func):
            self._method_converters.setdefault(op_name, {})[overload] = func
            return func

        return decorator

    def get_method_converter(self, op_name: str, overload: Optional[str] = None) -> Optional[Callable]:
        """获取方法和函数转换器"""
        if op_name in self._method_converters:
            table = self._method_converters[op_name]
            if overload:
                if overload in table:
                    return table[overload]
                else:
                    raise ValueError(f"Unsupported op.overload : {op_name}_{overload}")
            else:
                if None in table:
                    return table[None]
                else:
                    raise ValueError(f"Unsupported op.overload : {op_name}")
        else:
            raise ValueError(f"Unsupported op : {op_name}")


    def update(self, custom_converters: Dict):
        """更新转换器
        Args:
            custom_converters:
            {
                (op_name, overload): converter
            }
        """
        for key, converter in custom_converters.items():
            if isinstance(key, tuple) and len(key) == 2:
                op_name, overload = key
                self._method_converters.setdefault(op_name, {})[overload] = converter
            if isinstance(key, str):
                self._method_converters[key] = converter
            else:
                raise TypeError(f"Invalid key type: {type(key)}")

    def clear(self):
        """清空所有转换器"""
        self._method_converters.clear()

    def list_all_converters(self):
        """列出所有转换器"""
        return {
            "methods": list(self._method_converters.keys()),
        }


# 全局注册器实例
registry = ConverterRegistry()
