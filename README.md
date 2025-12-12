# InfiniTensor_v2.0

# InfiniTensor 格式化工具使用说明

本工具用于对项目中的 C/C++ 文件和 Python 文件进行统一的代码格式化，支持对 Git 已添加文件或指定分支间的修改文件进行格式化。  

---

## 支持文件类型

| 类型 | 文件后缀 |
|------|----------|
| C/C++ | `.h`, `.hh`, `.hpp`, `.c`, `.cc`, `.cpp`, `.cxx` |
| Python | `.py` |

---

## 使用方法

### 1. 格式化 Git 已添加文件（默认模式）

运行脚本而不传递任何参数时，工具会自动格式化当前 Git 仓库中已添加（`git add`）或修改（`git status` 显示 `modified:`）的文件：

```bash
python format.py
```

### 2. 格式化指定分支间的修改文件

运行脚本并传递分支名作为参数，工具会自动格式化指定分支间的修改文件：

```bash
python format.py <commit-id>
```

## 注意事项

### 工具依赖

- clang-format: 21版本，用于 C/C++ 文件格式化。
- black：用于 Python 文件格式化。
