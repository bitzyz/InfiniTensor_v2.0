#pragma once
#ifndef PYTHON_TENSOR_HPP
#define PYTHON_TENSOR_HPP
#include "core/runtime.h"
#include "core/tensor.h"
#include "dtype.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infini {

ShapeExpr create_shape_expr_from_pyobject(py::object dims) {
  std::vector<Expr> dim_exprs;
  if (py::isinstance<py::list>(dims) || py::isinstance<py::tuple>(dims)) {
    auto dim_list = dims.cast<py::list>();
    for (auto dim : dim_list) {
      if (py::isinstance<py::int_>(dim)) {
        int64_t value = dim.cast<int64_t>();
        dim_exprs.push_back(ExprObj::constant(value));
      } else if (py::isinstance<py::str>(dim)) {
        std::string name = dim.cast<std::string>();
        dim_exprs.push_back(ExprObj::variable(name));
      } else {
        throw py::type_error("Dimension must be int, str");
      }
    }
  } else {
    throw py::type_error("dims must be list, tuple");
  }
  return make_ref<ShapeExprObj>(dim_exprs);
}

StrideExpr create_stride_expr_from_pyobject(py::object strides) {
  std::vector<Expr> stride_exprs;
  if (py::isinstance<py::list>(strides) || py::isinstance<py::tuple>(strides)) {
    auto stride_list = strides.cast<py::list>();
    for (auto stride : stride_list) {
      if (py::isinstance<py::int_>(stride)) {
        int64_t value = stride.cast<int64_t>();
        stride_exprs.push_back(ExprObj::constant(value));
      } else if (py::isinstance<py::str>(stride)) {
        std::string name = stride.cast<std::string>();
        stride_exprs.push_back(ExprObj::variable(name));
      } else {
        throw py::type_error("Stride must be int, str");
      }
    }
  } else {
    throw py::type_error("strides must be list, tuple");
  }
  return make_ref<StrideExprObj>(stride_exprs);
}

void bind_tensor(py::module &m) {
  py::class_<ShapeExprObj, std::shared_ptr<ShapeExprObj>>(m, "ShapeExpr")
      .def(py::init<>())
      .def(py::init<>([](py::object dims) {
             return create_shape_expr_from_pyobject(dims);
           }),
           py::arg("dims"))
      .def("get_constant_value", &ShapeExprObj::getConstantValue)
      .def("to_string", &ShapeExprObj::toString);
  py::class_<StrideExprObj, std::shared_ptr<StrideExprObj>>(m, "StrideExpr")
      .def(py::init<>([](py::object strides) {
             return create_stride_expr_from_pyobject(strides);
           }),
           py::arg("strides"))
      .def("get_constant_value", &StrideExprObj::getConstantValue);
  py::class_<TensorObj, std::shared_ptr<TensorObj>>(m, "Tensor")
      .def("shape", &TensorObj::getShape)
      .def("dtype", &TensorObj::getDataType)
      .def("stride", &TensorObj::getStride)
      .def("rank", &TensorObj::getRank)
      .def("to_torch_info",
           [](TensorObj &self, Runtime &runtime) {
             if (!runtime->isCpu()) {
               self.copyToHost(runtime);
             }
             auto data_type = self.getDataType();
             auto shape = self.getShape()->getConstantValue();
             auto stride = self.getStride()->getConstantValue();
             void *data_ptr = self.getRawDataPtr<void *>();
             auto shape_vec = py::cast(shape);
             auto stride_vec = py::cast(stride);
             auto dtype_str = dtype_to_string(data_type);
             auto data_ptr_int = reinterpret_cast<uintptr_t>(data_ptr);
             return py::make_tuple(data_ptr_int,        // 数据指针
                                   shape_vec,           // 形状
                                   stride_vec,          // 步长
                                   dtype_str,           // 数据类型字符串
                                   self.getTotalBytes() // 存储大小
             );
           })
      .def("set_data",
           [](TensorObj &self, uintptr_t ptr, Runtime &runtime) {
             self.setData(reinterpret_cast<void *>(ptr));
             if (!runtime->isCpu()) {
               self.copyToDevice(runtime);
             }
           })
      .def("set_shape",
           [](TensorObj &self, py::object shape) {
             auto shape_expr = create_shape_expr_from_pyobject(shape);
             self.setShape(shape_expr);
           })
      .def("to_string", &TensorObj::toString);
}
} // namespace infini
#endif // PYTHON_TENSOR_HPP
