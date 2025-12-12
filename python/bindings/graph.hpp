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
        .def("to_string", &GraphBuilderObj::printGraph)
        .def_property_readonly("graph", &GraphBuilderObj::getGraph);
}

} // namespace infini
#endif // PYTHON_GRAPH_HPP
