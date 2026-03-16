# Tasking Plan - Iteration 2: Conv & LayerNorm Integration

## 目标
完成 `Conv` 和 `LayerNorm` 的全链路打通，实现“大满贯”算子添加目标。

## 任务拆解

### Story-9: InfiniCore 接口检查与 TDD 测试编写
- **描述**：明确 InfiniCore 底层对于 `Conv` 和 `LayerNorm` 提供的 API，并依据 TDD 编写 Python 端到端测试。
- **AC（验收标准）**：
  - 明确底层的 C-API 参数。
  - 完成 `test_conv.py` 和 `test_layernorm.py` 骨架及 `np.allclose` 断言。

### Story-10: 后端算子图与 Kernel 映射 (Conv & LayerNorm)
- **描述**：在 InfiniTensor_v2.0 C++ 层定义算子并桥接 `compute()`。
- **AC（验收标准）**：
  - 在 `src/operators/` 增加 `Conv` / `LayerNorm`，实现 shape/dtype inference。
  - 在 `src/kernels/` 调用 `infiniopConv` / `infiniopLayerNorm` 接口。

### Story-11: 前端 Pybind 绑定与 PyTorch FX 映射
- **描述**：将后端的 GraphBuilder 暴露给 Python，并在统一转换器中解析 `torch.nn.Conv2d` / `torch.nn.LayerNorm`。
- **AC（验收标准）**：
  - 更新 `python/bindings/` 暴露相应 `builder` 接口。
  - 在 `unified_converters.py` 中增加映射函数。

### Story-12: 端到端冒烟测试与修正
- **描述**：运行 pytest 跑通 `test_conv.py` 和 `test_layernorm.py`。
- **AC（验收标准）**：
  - 成功跑通所有单测，数值误差小于 `1e-4`。
  - 更新 `problems.log.md` 和 `progress.md`。

## 交付流与状态跟踪
- **进度与风险记录**：统一维护在 `report/progress.md` 中。
