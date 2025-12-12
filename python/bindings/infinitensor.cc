#include "dtype.hpp"
#include "graph.hpp"
#include "runtime.hpp"
#include "tensor.hpp"

namespace infini {

PYBIND11_MODULE(pyinfinitensor, m) {
    infini::bind_graph_builder(m);
    infini::bind_data_type(m);
    infini::bind_dtype_functions(m);
    infini::bind_tensor(m);
    infini::bind_runtime(m);
}

} // namespace infini
