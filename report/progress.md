# Project Progress & Risk Tracking

## 里程碑：算子添加模块 (Iteration 1)
**总目标**：在 `hotfix_20260316` 分支上完成目标核心算子的端到端接入、NVidia 高性能 Kernel 实现及多平台预留，通过冒烟测试。

### 进度追踪 (Progress)

| 编号 | Story 标题 | 状态 | 剩余工作量 (天) | 负责人 | 关联 AC / 备注 |
|---|---|---|---|---|---|
| Story-1 | InfiniCore 算子接口定义与底层注册 | 已完成 | 0 | Simon | C-API契约, 90%单测覆盖, clang-format |
| Story-2 | InfiniTensor_v2.0 算子接口定义与注册 | 已完成 | 0 | Simon | IR定义(.def), 形状/类型推导, 90%单测覆盖 |
| Story-3 | InfiniCore 后端 Kernel 实现 (RTX2060) | 已完成 | 0 | Simon | FP32/FP16, 精度达标, 性能基线≤110%, nsys报告 |
| Story-4 | InfiniTensor_v2.0 后端 Kernel 调用映射 | 已完成 | 0 | Simon | `compute()` 桥接, 精度透传, clang-format |
| Story-5 | InfiniCore 多平台架构预留 | 已完成 | 0 | Simon | muxi/tianshu/moore 预留, CMake BACKEND支持 |
| Story-6 | InfiniTensor_v2.0 单平台冒烟验证 | 已完成 | 0 | Simon | pytest, 静态/动态/边界shape, 无CUDA error |
| Story-7 | 前端 API 绑定与 PyTorch FX 映射 | 已完成 | 0 | Simon | FX转换对接, pybind绑定 |
| Story-8 | 端到端正确性验证与测试实化 | 已完成 | 0 | Simon | `np.allclose`误差对比 |
| Story-9 | InfiniCore 接口检查与 TDD 测试编写 | 已完成 | 0 | Simon | Conv & LayerNorm 的 C-API 确认与 Python TDD 测试 |
| Story-10 | 后端算子图与 Kernel 映射 (Conv & LayerNorm) | 已完成 | 0 | Simon | `src/operators` 与 `src/kernels` 对接 |
| Story-11 | 前端 Pybind 绑定与 PyTorch FX 映射 (Conv & LN) | 已完成 | 0 | Simon | pybind11 接口暴露与 `unified_converters.py` 支持 |
| Story-12 | 端到端冒烟测试与修正 (Conv & LN) | 已完成 | 0 | Simon | `np.allclose`误差对比 |
| Final | 完整冒烟测试与 `judge.md` 交付 | 已完成 | 0 | Simon | 大满贯达成 |

**当前总剩余工作量**：约 0 人日。

### 风险日志 (Risk Log)

| 日期 | 风险描述 | 影响面 | 缓解措施 (Mitigation) | 状态 |
|---|---|---|---|---|
| 2026-03-16 | 环境依赖：RTX2060 及对应 CUDA 11.8+cuDNN 8.x Docker 镜像的就绪情况可能影响 Story-3 和 Story-6 的验收。 | 阻塞 Kernel 性能调优与冒烟测试。 | 提前拉取验证 docker 镜像 `nvidia/cuda:11.8-devel-ubuntu22.04`，确认本地 GPU 驱动兼容性。 | 开放 |

---
*上次更新时间：2026-03-16*