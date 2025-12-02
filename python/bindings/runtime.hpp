#pragma once
#ifndef PYTHON_RUNTIME_HPP
#define PYTHON_RUNTIME_HPP
#include "core/runtime.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infini {
void bind_runtime(py::module &m) {
  py::enum_<infiniDevice_t>(m, "DeviceType")
      .value("CPU", INFINI_DEVICE_CPU)
      .value("CUDA", INFINI_DEVICE_NVIDIA)
      .export_values();
  py::class_<RuntimeObj, std::shared_ptr<RuntimeObj>>(m, "Runtime")
      .def(py::init<>())
      .def_static("get_instance", &RuntimeObj::getInstance,
                  py::return_value_policy::reference,
                  "Get the singleton runtime instance")
      .def_static("init", &RuntimeObj::init, "Initialize the runtime system")
      .def("init_thread_context", &RuntimeObj::initThreadContext,
           py::arg("device") = INFINI_DEVICE_CPU, py::arg("device_id") = 0,
           "Initialize thread context for current thread")
      .def(
          "run",
          [](RuntimeObj &self, Graph &graph) {
            self.dataMalloc(graph);
            self.run(graph);
          },
          py::arg("graph"), "Run computation graph");
}
} // namespace infini
#endif // PYTHON_RUNTIME_HPP
