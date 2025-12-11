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
      .value("MLU", INFINI_DEVICE_CAMBRICON)
      .value("ASCEND", INFINI_DEVICE_ASCEND)
      .value("METAX", INFINI_DEVICE_METAX)
      .value("MOORE", INFINI_DEVICE_MOORE)
      .value("ILUVATAR", INFINI_DEVICE_ILUVATAR)
      .value("KUNLUN", INFINI_DEVICE_KUNLUN)
      .value("HYGON", INFINI_DEVICE_HYGON)
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
      .def_static(
          "setup",
          [](infiniDevice_t device, int device_id) {
            RuntimeObj::init();
            auto runtime = RuntimeObj::getInstance();
            runtime->initThreadContext(device, device_id);
            return runtime;
          },
          py::arg("device") = INFINI_DEVICE_CPU, py::arg("device_id") = 0,
          py::return_value_policy::reference,
          "Initialize runtime system and thread context, and return Runtime "
          "instance")
      .def(
          "run",
          [](RuntimeObj &self, Graph &graph) {
            graph->shape_infer();
            self.dataMalloc(graph);
            self.run(graph);
          },
          py::arg("graph"), "Run computation graph");
}
} // namespace infini
#endif // PYTHON_RUNTIME_HPP
