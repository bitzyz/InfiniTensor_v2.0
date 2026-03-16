# Tasking Plan - Iteration 3: Grand Slam Operator Integration

## 目标
完成剩余所有算子 (`Softmax`, `LogSoftmax`, `LpNorm`, `RMSNorm`, `UnaryOps`) 的集成，达成项目满星要求。

## 任务拆解

### Story-13: 算子添加大满贯

#### 1. Unary Ops Group (Relu, Sigmoid, Silu, Gelu, Softplus, Tanh)
- **TDD**: 编写 `test_unary.py`，使用 `@pytest.mark.parametrize` 覆盖所有一元算子。
- **Backend**:
  - 在 `OpType` 添加枚举。
  - 实现 `Unary.h/cc` (复用 `ElementWise` 或新建通用 Unary 模板)。
  - 实现 `UnaryKernel` (复用或宏定义生成)。
- **Frontend**:
  - `GraphBuilder` 暴露接口。
  - `unified_converters.py` 注册 `relu`, `sigmoid`, `silu`, `gelu`, `softplus`, `tanh`。

#### 2. Softmax & LogSoftmax
- **TDD**: 编写 `test_softmax.py`。
- **Backend**:
  - 实现 `Softmax.h/cc` (包含 axis 参数)。
  - 实现 `SoftmaxKernel`。
- **Frontend**:
  - 绑定与转换支持。

#### 3. RMSNorm
- **TDD**: 编写 `test_rmsnorm.py`。
- **Backend**:
  - 实现 `RMSNorm.h/cc`。
  - 实现 `RMSNormKernel`。
- **Frontend**:
  - 绑定与转换支持 (注意 `T5LayerNorm` 或自定义实现映射)。

#### 4. LpNorm
- **TDD**: 编写 `test_lpnorm.py`。
- **Backend**:
  - 实现 `LpNorm.h/cc` (p, dim, keepdim)。
  - 实现 `LpNormKernel`。
- **Frontend**:
  - 绑定与转换支持。

#### 5. 验证与交付
- 运行所有测试。
- 更新 `progress.md` 和 `judge.md`。
