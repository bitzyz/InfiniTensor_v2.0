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

void GraphBuilderObj::set_outputs(const TensorVec &outputs) {
    g->setOutputs(outputs);
}

void GraphBuilderObj::optimize() { g->optimize(); }
Tensor GraphBuilderObj::conv(Tensor x, Tensor w, Tensor bias,
                             vector<int64_t> pads, vector<int64_t> strides,
                             vector<int64_t> dilations,
                             std::optional<Tensor> y) {
    if (y.has_value()) {
        g->addOpWithOutputs<ConvObj>(std::move(x), std::move(w),
                                     std::move(bias), std::move(y.value()),
                                     pads, strides, dilations);
        return y.value();
    } else {
        return g
            ->addOp<ConvObj>(std::move(x), std::move(w), std::move(bias),
                             nullptr, pads, strides, dilations)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::clip(Tensor input, Tensor min_val, Tensor max_val,
                             std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<ClipObj>(std::move(input), std::move(min_val),
                                     std::move(max_val),
                                     std::move(output.value()));
        return output.value();
    } else {
        return g
            ->addOp<ClipObj>(std::move(input), std::move(min_val),
                             std::move(max_val), nullptr)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::softmax(Tensor input, int axis,
                                std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<SoftmaxObj>(std::move(input),
                                        std::move(output.value()), axis);
        return output.value();
    } else {
        return g->addOp<SoftmaxObj>(std::move(input), nullptr, axis)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::log_softmax(Tensor input,
                                    std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<LogSoftmaxObj>(std::move(input),
                                           std::move(output.value()));
        return output.value();
    } else {
        return g->addOp<LogSoftmaxObj>(std::move(input), nullptr)->getOutput(0);
    }
}

Tensor GraphBuilderObj::layer_norm(Tensor input, Tensor weight, Tensor bias,
                                   std::optional<Tensor> output, int axis,
                                   float eps) {
    if (output.has_value()) {
        g->addOpWithOutputs<LayerNormObj>(std::move(input), std::move(weight),
                                          std::move(bias),
                                          std::move(output.value()), axis, eps);
        return output.value();
    } else {
        return g
            ->addOp<LayerNormObj>(std::move(input), std::move(weight),
                                  std::move(bias), nullptr, axis, eps)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::rms_norm(Tensor input, Tensor weight,
                                 std::optional<Tensor> output, float epsilon) {
    if (output.has_value()) {
        g->addOpWithOutputs<RMSNormObj>(std::move(input), std::move(weight),
                                        std::move(output.value()), epsilon);
        return output.value();
    } else {
        return g
            ->addOp<RMSNormObj>(std::move(input), std::move(weight), nullptr,
                                epsilon)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::lp_norm(Tensor input, int axis, int p,
                                std::optional<Tensor> output, float eps) {
    if (output.has_value()) {
        g->addOpWithOutputs<LPNormObj>(std::move(input),
                                       std::move(output.value()), axis, p, eps);
        return output.value();
    } else {
        return g->addOp<LPNormObj>(std::move(input), nullptr, axis, p, eps)
            ->getOutput(0);
    }
}

#define DEFINE_UNARY_OP(METHOD, TYPE)                                          \
    Tensor GraphBuilderObj::METHOD(Tensor input, std::optional<Tensor> Y) {    \
        if (Y.has_value()) {                                                   \
            g->addOpWithOutputs<UnaryObj>(TYPE, std::move(input),              \
                                          std::move(Y.value()));               \
            return Y.value();                                                  \
        } else {                                                               \
            return g->addOp<UnaryObj>(TYPE, std::move(input), nullptr)         \
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
