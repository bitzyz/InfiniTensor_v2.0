#pragma once
#ifndef PYTHON_DTYPE_HPP
#define PYTHON_DTYPE_HPP
#include "core/dtype.h"
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <torch/torch.h>

namespace py = pybind11;

namespace infini {
void bind_data_type(py::module &m) {
    py::class_<DataType>(m, "DType")
        // 构造函数
        .def(py::init<infiniDtype_t>())

        // 方法
        .def("get_size", &DataType::getSize, "Get the size in bytes")
        .def("get_type", &DataType::getType, "Get the underlying enum type")
        .def("to_string", &DataType::toString, "Get string representation")

        // 属性（通过方法暴露）
        .def_property_readonly("size", &DataType::getSize)
        .def_property_readonly("type", &DataType::getType)
        .def_property_readonly("name", &DataType::toString)

        // 运算符重载
        .def(py::self == py::self)
        .def(py::self != py::self);
}
inline DataType dtype_from_string(const std::string &dtype_str) {
    // 从字符串创建DataType
    static std::unordered_map<std::string, infiniDtype_t> str_to_dtype = {
        {"byte", INFINI_DTYPE_BYTE},
        {"bool", INFINI_DTYPE_BOOL},
        {"torch.bool", INFINI_DTYPE_BOOL},
        {"int8", INFINI_DTYPE_I8},
        {"torch.int8", INFINI_DTYPE_I8},
        {"int16", INFINI_DTYPE_I16},
        {"torch.int16", INFINI_DTYPE_I16},
        {"torch.short", INFINI_DTYPE_I16},
        {"int32", INFINI_DTYPE_I32},
        {"torch.int32", INFINI_DTYPE_I32},
        {"torch.int", INFINI_DTYPE_I32},
        {"int64", INFINI_DTYPE_I64},
        {"torch.int64", INFINI_DTYPE_I64},
        {"torch.long", INFINI_DTYPE_I64},
        {"uint8", INFINI_DTYPE_U8},
        {"torch.uint8", INFINI_DTYPE_U8},
        {"uint16", INFINI_DTYPE_U16},
        {"torch.uint16", INFINI_DTYPE_U16},
        {"uint32", INFINI_DTYPE_U32},
        {"torch.uint32", INFINI_DTYPE_U16},
        {"uint64", INFINI_DTYPE_U64},
        {"torch.uint64", INFINI_DTYPE_U64},
        {"fp8", INFINI_DTYPE_F8},
        {"float16", INFINI_DTYPE_F16},
        {"torch.float16", INFINI_DTYPE_F16},
        {"torch.half", INFINI_DTYPE_F16},
        {"float32", INFINI_DTYPE_F32},
        {"torch.float32", INFINI_DTYPE_F32},
        {"torch.float", INFINI_DTYPE_F32},
        {"double", INFINI_DTYPE_F64},
        {"float64", INFINI_DTYPE_F64},
        {"torch.float64", INFINI_DTYPE_F64},
        {"torch.double", INFINI_DTYPE_F64},
        {"complex32", INFINI_DTYPE_C32},
        {"torch.complex32", INFINI_DTYPE_C32},
        {"torch.chalf", INFINI_DTYPE_C64},
        {"complex64", INFINI_DTYPE_C64},
        {"torch.complex64", INFINI_DTYPE_C64},
        {"torch.cfloat", INFINI_DTYPE_C64},
        {"complex128", INFINI_DTYPE_C128},
        {"torch.complex128", INFINI_DTYPE_C128},
        {"torch.cdouble", INFINI_DTYPE_C128},
        {"bfloat16", INFINI_DTYPE_BF16}};

    auto it = str_to_dtype.find(dtype_str);
    if (it != str_to_dtype.end()) {
        return DataType(it->second);
    }
    throw std::runtime_error("Unknown data type: " + dtype_str);
}

inline std::string dtype_to_string(const DataType &dtype) {
    // 从DataType创建字符串
    static std::unordered_map<infiniDtype_t, std::string> dtype_to_str = {
        {INFINI_DTYPE_BYTE, "byte"},       {INFINI_DTYPE_BOOL, "bool"},
        {INFINI_DTYPE_I8, "int8"},         {INFINI_DTYPE_I16, "int16"},
        {INFINI_DTYPE_I32, "int32"},       {INFINI_DTYPE_I64, "int64"},
        {INFINI_DTYPE_U8, "uint8"},        {INFINI_DTYPE_U16, "uint16"},
        {INFINI_DTYPE_U32, "uint32"},      {INFINI_DTYPE_U64, "uint64"},
        {INFINI_DTYPE_F8, "fp8"},          {INFINI_DTYPE_F16, "float16"},
        {INFINI_DTYPE_F32, "float32"},     {INFINI_DTYPE_F64, "float64"},
        {INFINI_DTYPE_C32, "complex32"},   {INFINI_DTYPE_C64, "complex64"},
        {INFINI_DTYPE_C128, "complex128"}, {INFINI_DTYPE_BF16, "bfloat16"}};

    auto it = dtype_to_str.find(dtype.getType());
    if (it != dtype_to_str.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown data type: " +
                             std::to_string(dtype.getType()));
}

// inline torch::ScalarType dtype_to_torch_scalar_type(const DataType &dtype) {
//   static std::unordered_map<infiniDtype_t, torch::ScalarType> dtype_to_torch{
//       {INFINI_DTYPE_BOOL, torch::kBool},    {INFINI_DTYPE_I8, torch::kInt8},
//       {INFINI_DTYPE_I16, torch::kInt16},    {INFINI_DTYPE_I32,
//       torch::kInt32}, {INFINI_DTYPE_I64, torch::kInt64},    {INFINI_DTYPE_U8,
//       torch::kUInt8}, {INFINI_DTYPE_U16, torch::kUInt16}, {INFINI_DTYPE_U32,
//       torch::kUInt32}, {INFINI_DTYPE_U64, torch::kUInt64}, {INFINI_DTYPE_F16,
//       torch::kFloat16}, {INFINI_DTYPE_F32, torch::kFloat32},
//       {INFINI_DTYPE_F64, torch::kFloat64}, {INFINI_DTYPE_BF16,
//       torch::kBFloat16}};
//   auto it = dtype_to_torch.find(dtype.getType());
//   if (it != dtype_to_torch.end()) {
//     return it->second;
//   }
//   throw std::runtime_error("Unsupported Convert DataType to torch: " +
//                            dtype.toString());
// }

void bind_dtype_functions(py::module &m) {
    m.def("dtype_from_string", &dtype_from_string,
          "Create DataType from string", py::arg("dtype_str"))
        .def("dtype_to_string", &dtype_to_string, "Convert DataType to string",
             py::arg("dtype"));
    // m.def("dtype_to_torch_scalar_type", &dtype_to_torch_scalar_type,
    //       "Convert DataType to torch::ScalarType", py::arg("dtype"));
}
} // namespace infini

#endif // PYTHON_DTYPE_HPP
