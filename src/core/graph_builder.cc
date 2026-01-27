#include "core/graph_builder.h"

namespace infini {

GraphBuilderObj::GraphBuilderObj(Runtime runtime)
    : g(make_ref<GraphObj>(std::move(runtime))) {}

Tensor GraphBuilderObj::tensor(ShapeExpr dims, DataType dtype,
                               std::optional<StrideExpr> stride) {
    if (stride.has_value()) {
        return g->addTensor(dims, stride.value(), dtype);
    } else {
        return g->addTensor(dims, dtype);
    }
}

Tensor GraphBuilderObj::gemm(Tensor A, Tensor B, Tensor C, float alpha,
                             float beta, bool transA, bool transB,
                             std::optional<Tensor> Y) {
    if (Y.has_value()) {
        g->addOpWithOutputs<GemmObj>(std::move(A), std::move(B),
                                     std::move(Y.value()), std::move(C), alpha,
                                     beta, transA, transB);
        return Y.value();
    } else {
        return g
            ->addOp<GemmObj>(std::move(A), std::move(B), nullptr, std::move(C),
                             alpha, beta, transA, transB)
            ->getOutput(0);
    }
}

#define DEFINE_BINARY_OP(OP, TYPE)                                             \
    Tensor GraphBuilderObj::OP(Tensor A, Tensor B, std::optional<Tensor> Y) {  \
        if (Y.has_value()) {                                                   \
            g->addOpWithOutputs<ElementWiseObj>(                               \
                TYPE, std::move(A), std::move(B), std::move(Y.value()));       \
            return Y.value();                                                  \
        } else {                                                               \
            return g                                                           \
                ->addOp<ElementWiseObj>(TYPE, std::move(A), std::move(B),      \
                                        nullptr)                               \
                ->getOutput(0);                                                \
        }                                                                      \
    }

DEFINE_BINARY_OP(add, OpType::Add);
DEFINE_BINARY_OP(sub, OpType::Sub);
DEFINE_BINARY_OP(mul, OpType::Mul);

string GraphBuilderObj::printGraph() const { return g->toString(); }

Graph GraphBuilderObj::getGraph() const { return g; }
} // namespace infini
