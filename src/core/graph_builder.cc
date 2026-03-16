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

Tensor GraphBuilderObj::clip(Tensor input, Tensor min, Tensor max,
                             std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<ElementWiseObj>(OpType::Clip, std::move(input),
                                            std::move(min), std::move(max),
                                            std::move(output.value()));
        return output.value();
    } else {
        return g
            ->addOp<ElementWiseObj>(OpType::Clip, std::move(input),
                                    std::move(min), std::move(max), nullptr)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::conv(Tensor input, Tensor weight, std::optional<Tensor> bias,
                             std::vector<int> pads, std::vector<int> strides,
                             std::vector<int> dilations, std::optional<Tensor> output) {
    Tensor b = bias.has_value() ? bias.value() : nullptr;
    if (output.has_value()) {
        g->addOpWithOutputs<ConvObj>(std::move(input), std::move(weight),
                                     std::move(output.value()), std::move(pads),
                                     std::move(strides), std::move(dilations), std::move(b));
        return output.value();
    } else {
        return g->addOp<ConvObj>(std::move(input), std::move(weight), nullptr,
                                 std::move(pads), std::move(strides), std::move(dilations), std::move(b))
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::layer_norm(Tensor input, Tensor weight, Tensor bias, float eps,
                                   std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<LayerNormObj>(std::move(input), std::move(weight), std::move(bias),
                                          std::move(output.value()), eps);
        return output.value();
    } else {
        return g->addOp<LayerNormObj>(std::move(input), std::move(weight), std::move(bias), nullptr, eps)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::relu(Tensor input, std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<ReluObj>(std::move(input), std::move(output.value()));
        return output.value();
    } else {
        return g->addOp<ReluObj>(std::move(input), nullptr)->getOutput(0);
    }
}

Tensor GraphBuilderObj::sigmoid(Tensor input, std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<SigmoidObj>(std::move(input), std::move(output.value()));
        return output.value();
    } else {
        return g->addOp<SigmoidObj>(std::move(input), nullptr)->getOutput(0);
    }
}

Tensor GraphBuilderObj::tanh(Tensor input, std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<TanhObj>(std::move(input), std::move(output.value()));
        return output.value();
    } else {
        return g->addOp<TanhObj>(std::move(input), nullptr)->getOutput(0);
    }
}

Tensor GraphBuilderObj::gelu(Tensor input, std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<GeluObj>(std::move(input), std::move(output.value()));
        return output.value();
    } else {
        return g->addOp<GeluObj>(std::move(input), nullptr)->getOutput(0);
    }
}

Tensor GraphBuilderObj::silu(Tensor input, std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<SiluObj>(std::move(input), std::move(output.value()));
        return output.value();
    } else {
        return g->addOp<SiluObj>(std::move(input), nullptr)->getOutput(0);
    }
}

Tensor GraphBuilderObj::softplus(Tensor input, std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<SoftplusObj>(std::move(input), std::move(output.value()));
        return output.value();
    } else {
        return g->addOp<SoftplusObj>(std::move(input), nullptr)->getOutput(0);
    }
}

Tensor GraphBuilderObj::softmax(Tensor input, int axis, std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<SoftmaxObj>(std::move(input), std::move(output.value()), axis);
        return output.value();
    } else {
        return g->addOp<SoftmaxObj>(std::move(input), nullptr, axis)->getOutput(0);
    }
}

Tensor GraphBuilderObj::log_softmax(Tensor input, int axis, std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<LogSoftmaxObj>(std::move(input), std::move(output.value()), axis);
        return output.value();
    } else {
        return g->addOp<LogSoftmaxObj>(std::move(input), nullptr, axis)->getOutput(0);
    }
}

Tensor GraphBuilderObj::rms_norm(Tensor input, Tensor weight, float eps, std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<RMSNormObj>(std::move(input), std::move(weight), std::move(output.value()), eps);
        return output.value();
    } else {
        return g->addOp<RMSNormObj>(std::move(input), std::move(weight), nullptr, eps)->getOutput(0);
    }
}

Tensor GraphBuilderObj::lp_norm(Tensor input, float p, std::vector<int> dims, bool keepdim, std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<LpNormObj>(std::move(input), std::move(output.value()), p, std::move(dims), keepdim);
        return output.value();
    } else {
        return g->addOp<LpNormObj>(std::move(input), nullptr, p, std::move(dims), keepdim)->getOutput(0);
    }
}

Tensor GraphBuilderObj::transpose(Tensor input, std::vector<int> perm, std::optional<Tensor> output) {
    if (output.has_value()) {
        g->addOpWithOutputs<TransposeObj>(std::move(input), std::move(output.value()), std::move(perm));
        return output.value();
    } else {
        return g->addOp<TransposeObj>(std::move(input), nullptr, std::move(perm))->getOutput(0);
    }
}

string GraphBuilderObj::printGraph() const { return g->toString(); }

Graph GraphBuilderObj::getGraph() const { return g; }
} // namespace infini
