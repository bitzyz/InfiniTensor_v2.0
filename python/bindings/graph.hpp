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
        .def("clamp", &GraphBuilderObj::clip, py::arg("IN"), py::arg("MIN"),
             py::arg("MAX"), py::arg("Y") = py::none())
        .def("conv", &GraphBuilderObj::conv, py::arg("x"), py::arg("weight"),
             py::arg("bias"), py::arg("stride"), py::arg("padding"),
             py::arg("dilation"), py::arg("n"), py::arg("Y") = py::none())
        .def("layer_norm", &GraphBuilderObj::layer_norm, py::arg("x"),
             py::arg("weight"), py::arg("bias"), py::arg("eps"),
             py::arg("Y") = py::none(), py::arg("Norm") = py::none(),
             py::arg("Std") = py::none())
        .def("log_softmax", &GraphBuilderObj::log_softmax, py::arg("x"),
             py::arg("dim") = 1, py::arg("Y") = py::none())
        .def("softmax", &GraphBuilderObj::softmax, py::arg("x"),
             py::arg("axis") = 1, py::arg("Y") = py::none())
        .def("lp_norm", &GraphBuilderObj::lp_norm, py::arg("x"),
             py::arg("axis") = 1, py::arg("p") = 2, py::arg("eps") = 1e-12f,
             py::arg("Y") = py::none())
        .def("rms_norm", &GraphBuilderObj::rms_norm, py::arg("x"),
             py::arg("weight"), py::arg("eps") = 1e-5f,
             py::arg("Y") = py::none())
        .def("relu", &GraphBuilderObj::relu, py::arg("x"),
             py::arg("Y") = py::none())
        .def("sigmoid", &GraphBuilderObj::sigmoid, py::arg("x"),
             py::arg("Y") = py::none())
        .def("silu", &GraphBuilderObj::silu, py::arg("x"),
             py::arg("Y") = py::none())
        .def("gelu", &GraphBuilderObj::gelu, py::arg("x"),
             py::arg("Y") = py::none())
        .def("softplus", &GraphBuilderObj::softplus, py::arg("x"),
             py::arg("Y") = py::none())
        .def("tanh", &GraphBuilderObj::tanh, py::arg("x"),
             py::arg("Y") = py::none())
        .def("to_string", &GraphBuilderObj::printGraph)
        .def_property_readonly("graph", &GraphBuilderObj::getGraph);
}

} // namespace infini
#endif // PYTHON_GRAPH_HPP
