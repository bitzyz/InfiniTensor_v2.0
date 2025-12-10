#pragma once
#ifndef PYTHON_TENSOR_HPP
#define PYTHON_TENSOR_HPP
#include "core/runtime.h"
#include "core/tensor.h"
#include "dtype.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <torch/torch.h>

namespace py = pybind11;

namespace infini {
void bind_tensor(py::module &m) {
  py::class_<TensorObj, std::shared_ptr<TensorObj>>(m, "Tensor")
      .def("shape", &TensorObj::getShape)
      .def("dtype", &TensorObj::getDataType)
      .def("stride", &TensorObj::getStride)
      .def("rank", &TensorObj::getRank)
      //   .def("from_torch_zero_copy",
      //        [](TensorObj &self, torch::Tensor torch_tensor) {
      //          // 检查形状是否匹配
      //          auto torch_shape = torch_tensor.sizes();
      //          auto self_shape = self.getShape();

      //          if (torch_shape.size() != self_shape.size()) {
      //            throw std::runtime_error("Shape rank mismatch between
      //            existing "
      //                                     "tensor and PyTorch tensor");
      //          }

      //          for (size_t i = 0; i < torch_shape.size(); ++i) {
      //            if (torch_shape[i] != self_shape[i]) {
      //              throw std::runtime_error("Shape dimension mismatch between
      //              "
      //                                       "existing tensor and PyTorch
      //                                       tensor");
      //            }
      //          }

      //          // 检查数据类型是否匹配
      //          DataType self_dtype = self.getDataType();
      //          py::object py_tensor = py::cast(torch_tensor);
      //          py::object dtype_attr = py_tensor.attr("dtype");
      //          py::object dtype_str_attr = dtype_attr.attr("__str__")();
      //          std::string dtype_str = py::cast<std::string>(dtype_str_attr);
      //          DataType torch_dtype = dtype_from_string(dtype_str);
      //          if (self_dtype.getIndex() != expected_dtype.getIndex()) {
      //            throw std::runtime_error("Data type mismatch between
      //            existing "
      //                                     "tensor and PyTorch tensor");
      //          }

      //          // 直接设置数据指针（零拷贝）
      //          void *data_ptr = torch_tensor.data_ptr();
      //          self.setData(data_ptr);

      //          return &self;
      //        })
      //   .def("to_torch_tensor",
      //        [](TensorObj &self, RuntimeObj &runtime) {
      //          // 先拷贝到CPU
      //          auto tensor = self.copyToCpu();
      //          auto torch_dtype =
      //              dtype_to_torch_scalar_type(tensor.getDataType());
      //          auto shape = tensor.getShape();
      //          auto stride = tensor.getStride();
      //          auto options =
      //              torch::TensorOptions().dtype(torch_dtype).device(torch::kCPU);
      //          auto torch_tensor = torch::from_blob(
      //              tensor.getRawDataPtr<void *>(), shape, stride, options);
      //          return torch_tensor;
      //        })
      .def("to_torch_info",
           [](TensorObj &self, Runtime &runtime) {
             if (!runtime->isCpu()) {
               self.copyToHost(runtime);
             }
             auto data_type = self.getDataType();
             auto shape = self.getShape();
             auto stride = self.getStride();
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
      .def("set_data", [](TensorObj &self, uintptr_t ptr, Runtime &runtime) {
        self.setData(reinterpret_cast<void *>(ptr));
        if (!runtime->isCpu()) {
          self.copyToDevice(runtime);
        }
      });
}
} // namespace infini
#endif // PYTHON_TENSOR_HPP
