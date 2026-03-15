#include "core/graph_builder.h"

#include "operators/UnaryOp.h"

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

Tensor GraphBuilderObj::clip(Tensor IN, Tensor MIN, Tensor MAX,
                             std::optional<Tensor> Y) {
    if (Y.has_value()) {
        g->addOpWithOutputs<ClipObj>(std::move(IN), std::move(Y.value()),
                                     std::move(MAX), std::move(MIN));
        return Y.value();
    } else {
        return g
            ->addOp<ClipObj>(std::move(IN), nullptr, std::move(MAX),
                             std::move(MIN))
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::conv(Tensor x, Tensor weight, Tensor bias,
                             const std::vector<size_t> &strides,
                             const std::vector<size_t> &paddings,
                             const std::vector<size_t> &dilations, int n,
                             std::optional<Tensor> Y) {
    if (Y.has_value()) {
        auto result = Y.value();
        g->addOpWithOutputs<ConvObj>(std::move(x), std::move(weight),
                                     std::move(bias), strides, dilations,
                                     paddings, std::move(Y.value()));
        return result;
    } else {
        return g
            ->addOp<ConvObj>(std::move(x), std::move(weight), std::move(bias),
                             strides, dilations, paddings, nullptr)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::layer_norm(Tensor x, Tensor weight, Tensor bias,
                                   float eps, std::optional<Tensor> Y,
                                   std::optional<Tensor> Norm,
                                   std::optional<Tensor> Std) {
    if (Y.has_value()) {
        auto result = Y.value();
        g->addOpWithOutputs<LayerNormObj>(
            std::move(x), std::move(weight), std::move(bias), eps,
            std::move(Y.value()), std::move(Norm.value()),
            std::move(Std.value()));
        return result;
    } else {
        return g
            ->addOp<LayerNormObj>(std::move(x), std::move(weight),
                                  std::move(bias), eps, nullptr, nullptr,
                                  nullptr)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::log_softmax(Tensor x, int dim,
                                    std::optional<Tensor> Y) {
    if (Y.has_value()) {
        g->addOpWithOutputs<LogSoftmaxObj>(std::move(x), dim,
                                           std::move(Y.value()));
        return Y.value();
    } else {
        return g->addOp<LogSoftmaxObj>(std::move(x), dim, nullptr)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::softmax(Tensor x, int axis, std::optional<Tensor> Y) {
    if (Y.has_value()) {
        g->addOpWithOutputs<SoftmaxObj>(std::move(x), axis,
                                        std::move(Y.value()));
        return Y.value();
    } else {
        return g->addOp<SoftmaxObj>(std::move(x), axis, nullptr)->getOutput(0);
    }
}

Tensor GraphBuilderObj::lp_norm(Tensor x, int axis, int p, float eps,
                                std::optional<Tensor> Y) {
    if (Y.has_value()) {
        g->addOpWithOutputs<LpNormObj>(std::move(x), axis, p, eps,
                                       std::move(Y.value()));
        return Y.value();
    } else {
        return g->addOp<LpNormObj>(std::move(x), axis, p, eps, nullptr)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::rms_norm(Tensor x, Tensor weight, float eps,
                                 std::optional<Tensor> Y) {
    if (Y.has_value()) {
        g->addOpWithOutputs<RmsNormObj>(std::move(x), std::move(weight), eps,
                                        std::move(Y.value()));
        return Y.value();
    } else {
        return g
            ->addOp<RmsNormObj>(std::move(x), std::move(weight), eps, nullptr)
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

#define DEFINE_UNARY_OP(OP, TYPE)                                              \
    Tensor GraphBuilderObj::OP(Tensor X, std::optional<Tensor> Y) {            \
        if (Y.has_value()) {                                                   \
            g->addOpWithOutputs<UnaryWiseObj>(TYPE, std::move(X),              \
                                              std::move(Y.value()));           \
            return Y.value();                                                  \
        } else {                                                               \
            return g->addOp<UnaryWiseObj>(TYPE, std::move(X), nullptr)         \
                ->getOutput(0);                                                \
        }                                                                      \
    }

DEFINE_UNARY_OP(relu, OpType::Relu)
DEFINE_UNARY_OP(sigmoid, OpType::Sigmoid)
DEFINE_UNARY_OP(silu, OpType::Silu)
DEFINE_UNARY_OP(gelu, OpType::Gelu)
DEFINE_UNARY_OP(softplus, OpType::Softplus)
DEFINE_UNARY_OP(tanh, OpType::Tanh)

string GraphBuilderObj::printGraph() const { return g->toString(); }

Graph GraphBuilderObj::getGraph() const { return g; }
} // namespace infini
