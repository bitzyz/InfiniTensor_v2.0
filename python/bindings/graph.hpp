#pragma once
#ifndef PYTHON_GRAPH_HPP
#define PYTHON_GRAPH_HPP
#include "core/graph_builder.h"
#include "core/runtime.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infini {
void bind_graph_builder(py::module &m) {
    py::class_<GraphObj, std::shared_ptr<GraphObj>>(m, "Graph");
    // GraphBuilder
    py::class_<GraphBuilderObj>(m, "GraphBuilder")
        .def(py::init<Runtime>())
        .def("tensor", &GraphBuilderObj::tensor, py::arg("dims"),
             py::arg("dtype"), py::arg("stride") = py::none())
        .def("gemm", &GraphBuilderObj::gemm, py::arg("A"), py::arg("B"),
             py::arg("C"), py::arg("alpha") = 1.0, py::arg("beta") = 1.0,
             py::arg("transA") = false, py::arg("transB") = false,
             py::arg("Y") = py::none())
        .def("add", &GraphBuilderObj::add, py::arg("A"), py::arg("B"),
             py::arg("Y") = py::none())
        .def("sub", &GraphBuilderObj::sub, py::arg("A"), py::arg("B"),
             py::arg("Y") = py::none())
        .def("mul", &GraphBuilderObj::mul, py::arg("A"), py::arg("B"),
             py::arg("Y") = py::none())
        .def("clip", &GraphBuilderObj::clip, py::arg("input"), py::arg("min"),
             py::arg("max"), py::arg("output") = py::none())
        .def("conv", &GraphBuilderObj::conv, py::arg("input"), py::arg("weight"),
             py::arg("bias") = py::none(), py::arg("pads"), py::arg("strides"),
             py::arg("dilations"), py::arg("output") = py::none())
        .def("layer_norm", &GraphBuilderObj::layer_norm, py::arg("input"), py::arg("weight"),
             py::arg("bias"), py::arg("eps") = 1e-5, py::arg("output") = py::none())
        .def("relu", &GraphBuilderObj::relu, py::arg("input"), py::arg("output") = py::none())
        .def("sigmoid", &GraphBuilderObj::sigmoid, py::arg("input"), py::arg("output") = py::none())
        .def("tanh", &GraphBuilderObj::tanh, py::arg("input"), py::arg("output") = py::none())
        .def("gelu", &GraphBuilderObj::gelu, py::arg("input"), py::arg("output") = py::none())
        .def("silu", &GraphBuilderObj::silu, py::arg("input"), py::arg("output") = py::none())
        .def("softplus", &GraphBuilderObj::softplus, py::arg("input"), py::arg("output") = py::none())
        .def("softmax", &GraphBuilderObj::softmax, py::arg("input"), py::arg("axis"), py::arg("output") = py::none())
        .def("log_softmax", &GraphBuilderObj::log_softmax, py::arg("input"), py::arg("axis"), py::arg("output") = py::none())
        .def("rms_norm", &GraphBuilderObj::rms_norm, py::arg("input"), py::arg("weight"), py::arg("eps") = 1e-6, py::arg("output") = py::none())
        .def("lp_norm", &GraphBuilderObj::lp_norm, py::arg("input"), py::arg("p"), py::arg("dims"), py::arg("keepdim") = false, py::arg("output") = py::none())
        .def("transpose", &GraphBuilderObj::transpose, py::arg("input"), py::arg("perm"), py::arg("output") = py::none())
        .def("to_string", &GraphBuilderObj::printGraph)
        .def_property_readonly("graph", &GraphBuilderObj::getGraph);
}

} // namespace infini
#endif // PYTHON_GRAPH_HPP
