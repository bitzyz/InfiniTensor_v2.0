# Tasking Plan: Iteration 1 (算子添加)

## 任务背景
根据 `2025冬季训练营AI编译器方向项目题目.docx` 与 `算子添加示例.docx` 需求，优先聚焦**算子添加**模块。在 `hotfix_20260316` 分支上依次完成任务一、二、三的开发，严格遵循 `Conventional Commits` 规范。

## 目标算子
选择并聚焦完成某个具有代表性的核心算子（如 Clip, Conv, LayerNorm 或 UnaryOp 中的一个，具体将在代码实现时根据实际情况选择，如 Clip）。

## 任务拆解与 Tasking Stories

### Story-1: InfiniCore 算子接口定义与底层注册
- **描述**：为上层提供清晰、一致的底层算子 C-API 契约。
- **AC（验收标准）**：
  - 完成算子（如 `infiniopClip`）参数（输入/输出/属性）的底层 API 头文件声明（`include/infiniop/ops/xxx.h`）。
  - 配合上层完成底层执行器的映射挂载（CPU/NVidia路径）。
  - 代码通过 `clang-format` (Google 风格) 检查。
  - C++ 单元测试行覆盖率 ≥ 90%。
  - 产出对应的 `README.md`。
- **技术要点**：定义稳定的 ABI 接口，工作区（Workspace）大小查询接口返回 0 或实际大小。
- **依赖关系**：无
- **预计工期**：1 天

### Story-2: InfiniTensor_v2.0 算子接口定义与注册 (前端与IR层)
- **描述**：在编译器中间表示（IR）层确立算子原型，确保编译期能正确识别算子语义。
- **AC（验收标准）**：
  - 在编译器 IR 层完成算子原型的形式化定义（`.def`文件）。
  - 在算子注册表（CPU/NVidia 路径）中新增该算子入口。
  - 覆盖形状推导（Shape Inference）和类型推导（DataType Inference）的单元测试，覆盖率 ≥ 90%。
  - 代码通过 `clang-format` 检查。
  - 产出对应的 `README.md`。
- **技术要点**：OperatorObj 继承实现，推导逻辑正确处理边界。
- **依赖关系**：依赖 Story-1 提供的底层 C-API 头文件契约。
- **预计工期**：1.5 天

### Story-3: InfiniCore NVidia RTX2060 后端 Kernel 实现
- **描述**：实现对应 GPU kernel，要求高性能与高精度。
- **AC（验收标准）**：
  - 基于 CUDA 11.x + cuDNN 8.x 实现 GPU kernel，支持 FP32/FP16。
  - 内存访问对齐，合并读写，Warp 并行度 ≥ 32。
  - 计算误差：FP32 ≤ 1e-4，FP16 ≤ 1e-2。
  - 性能基线：RTX2060 上典型形状 (N=32, C=256, H=128, W=128) 执行时间 ≤ 原生 110%。
  - 产出 `.cu` 源文件、基准测试脚本、nsys 性能报告。
- **技术要点**：CUDA 核心编程，显存合并访存优化。
- **依赖关系**：依赖 Story-1。
- **预计工期**：2 天

### Story-4: InfiniTensor_v2.0 后端 Kernel 调用接口映射
- **描述**：打通 InfiniTensor 上层计算图与 InfiniCore 底层 CUDA 内核的物理执行通道。
- **AC（验收标准）**：
  - 上层 Kernel 接口（如 `compute()`）正确调用底层 `infiniopXXX` 执行函数。
  - 正确透传 FP32/FP16 精度标记。
  - 代码通过 `clang-format` 检查。
- **技术要点**：处理好 Workspace 内存分配与底层流（Stream）绑定。
- **依赖关系**：依赖 Story-2 和 Story-3。
- **预计工期**：1 天

### Story-5: InfiniCore 多平台架构预留 (3.b)
- **描述**：为未来接入国产算力芯片建立多平台扩展骨架。
- **AC（验收标准）**：
  - 在算子目录下预留 `muxi`、`tianshu`、`moore` 子目录及空实现模板（返回 `NotImplementedError` 或对应 C++ 错误码）。
  - CMakeLists.txt 新增 `BACKEND` 枚举值，支持 `-DBACKEND=xxx` 切换。
  - 提交目录结构图与平台抽象接口头文件。
- **技术要点**：CMake 条件编译宏设计。
- **依赖关系**：无严格依赖，可在 Story-1 之后随时进行。
- **预计工期**：1 天

### Story-6: InfiniTensor_v2.0 单平台冒烟验证 (3.a)
- **描述**：通过端到端单平台（NVidia RTX2060）冒烟测试保障鲁棒性。
- **AC（验收标准）**：
  - 编写 pytest 用例，覆盖静态 shape、动态 shape、边界 case（空 Tensor、0 维 Tensor）。
  - 在 Docker 镜像 `nvidia/cuda:11.8-devel-ubuntu22.04` 内一次性通过，无 CUDA error。
  - 输出 `task3_smoke_report.txt` 结果文件。
- **技术要点**：PyTorch 前端 API 映射测试，边界异常处理。
- **依赖关系**：依赖 Story-4。
- **预计工期**：1.5 天

### Story-7: 前端 API 绑定与 PyTorch FX 映射
- **描述**：实现 Clip 算子在 Python 层的暴露及 PyTorch 转换对接。
- **AC（验收标准）**：
  - 在 `infinitensor/converter` 目录下的统一转换器中添加 `clamp` / `clip` 的转换映射逻辑。
  - 提供正确的 Pybind 绑定。
- **技术要点**：FX 图节点遍历、参数提取、类型映射。
- **依赖关系**：依赖后端图层面算子的可用性。
- **预计工期**：1 天

### Story-8: 端到端正确性验证与测试实化
- **描述**：实化 `test_clip.py` 端到端正确性验证测试。
- **AC（验收标准）**：
  - 移除原先伪造的 "placeholder passed"。
  - 真实构建模型、前向传播、提取结果并进行 `np.allclose` 对比验证。
- **技术要点**：E2E 验证，误差边界计算。
- **依赖关系**：依赖 Story-7。
- **预计工期**：0.5 天

## 交付流与状态跟踪
- **进度与风险记录**：统一维护在 `report/progress.md` 中。
- **最终冒烟测试**：全部完成后更新根目录 `judge.md`。