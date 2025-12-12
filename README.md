# InfiniTensor_v2.0

# InfiniTensor 项目使用说明
本文档介绍如何使用项目提供的 `Makefile` 进行构建、测试、安装 Python 包等操作。支持多种硬件平台（CPU / CUDA / Ascend / Cambricon / 等）。

---

## 1. 📌 环境要求

在使用本项目前，请确保系统已安装：

- **CMake ≥ 3.22**
- **GCC / Clang**
- **Python 3.10+**
- **XMake**
- **HardWare Toolkit**（可选，若启用对应硬件后端）

---

## 2. 📂 Makefile 参数说明

### 构建类型
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TYPE` | Release | 构建类型，可设为 Debug |

### 测试开关
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TEST` | ON | 是否编译测试代码 |

### 平台选择（必须一致）
| PLATFORM 值 | 含义 | 自动打开的开关 |
|-------------|-------|----------------|
| CPU | 纯 CPU | — |
| CUDA | NVIDIA GPU | `USE_CUDA=ON` |
| ASCEND | 华为 Ascend | `USE_ASCEND=ON` |
| CAMBRICON | 寒武纪 MLU | `USE_CAMBRICON=ON` |
| METAX | 沐曦 MTX | `USE_METAX=ON` |
| MOORE | 摩尔线程 Moore  | `USE_MOORE=ON` |
| ILUVATAR | 天数智芯 | `USE_ILUVATAR=ON` |
| SUGON | 曙光 DCU | `USE_SUGON=ON` |
| KUNLUN | 百度昆仑 XPU | `USE_KUNLUN=ON` |

---

## 3. 🛠️ 基本构建

### 3.1 构建项目

```bash
make build PLATFORM=CUDA
```

### 3.2 安装Python包

```bash
make install-python PLATFORM=CUDA
```
### 3.3 运行测试

```bash
make test && make test-front
```

### 3.4 清理构建目录

```bash
make clean
```

## 4. 📦 InfiniTensor 格式化工具使用说明

本工具用于对项目中的 C/C++ 文件和 Python 文件进行统一的代码格式化，支持对 Git 已添加文件或指定分支间的修改文件进行格式化。  

---

### 支持文件类型

| 类型 | 文件后缀 |
|------|----------|
| C/C++ | `.h`, `.hh`, `.hpp`, `.c`, `.cc`, `.cpp`, `.cxx` |
| Python | `.py` |

---

### 使用方法

#### 1. 格式化 Git 已添加文件（默认模式）

运行脚本而不传递任何参数时，工具会自动格式化当前 Git 仓库中已添加（`git add`）或修改（`git status` 显示 `modified:`）的文件：

```bash
python format.py
```

#### 2. 格式化指定分支间的修改文件

运行脚本并传递分支名作为参数，工具会自动格式化指定分支间的修改文件：

```bash
python format.py <commit-id>
```

### 注意事项

#### 工具依赖

- clang-format: 21版本，用于 C/C++ 文件格式化。
- black：用于 Python 文件格式化。
