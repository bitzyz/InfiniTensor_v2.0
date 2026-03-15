#pragma once
#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

#include "core/graph.h"
#include "core/op_type.h"
#include "operators/Clip.h"
#include "operators/Conv.h"
#include "operators/ElementWise.h"
#include "operators/Gemm.h"
#include "operators/LayerNorm.h"
#include "operators/LogSoftmax.h"
#include "operators/LpNorm.h"
#include "operators/RmsNorm.h"
#include "operators/Softmax.h"

namespace infini {

class GraphBuilderObj {
  private:
    Ref<GraphObj> g;

  public:
    GraphBuilderObj(Runtime runtime);

    Tensor tensor(ShapeExpr dims, DataType dtype,
                  std::optional<StrideExpr> stride = std::nullopt);

    Tensor gemm(Tensor A, Tensor B, Tensor C, float alpha = 1.0,
                float beta = 1.0, bool transA = false, bool transB = false,
                std::optional<Tensor> Y = std::nullopt);
    Tensor add(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor sub(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor mul(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor clip(Tensor in, Tensor min_val, Tensor max,
                std::optional<Tensor> Y = std::nullopt);
    Tensor conv(Tensor x, Tensor weight, Tensor bias,
                const std::vector<size_t> &strides,
                const std::vector<size_t> &paddings,
                const std::vector<size_t> &dilations, int n,
                std::optional<Tensor> Y);
    Tensor layer_norm(Tensor x, Tensor weight, Tensor bias, float eps,
                      std::optional<Tensor> Y, std::optional<Tensor> Norm,
                      std::optional<Tensor> Std);
    Tensor log_softmax(Tensor x, int dim,
                       std::optional<Tensor> Y = std::nullopt);
    Tensor softmax(Tensor x, int axis, std::optional<Tensor> Y = std::nullopt);
    Tensor lp_norm(Tensor x, int axis, int p, float eps,
                   std::optional<Tensor> Y = std::nullopt);
    Tensor rms_norm(Tensor x, Tensor weight, float eps,
                    std::optional<Tensor> Y = std::nullopt);

    Tensor relu(Tensor A, std::optional<Tensor> Y = std::nullopt);
    Tensor sigmoid(Tensor A, std::optional<Tensor> Y = std::nullopt);
    Tensor silu(Tensor A, std::optional<Tensor> Y = std::nullopt);
    Tensor gelu(Tensor A, std::optional<Tensor> Y = std::nullopt);
    Tensor softplus(Tensor A, std::optional<Tensor> Y = std::nullopt);
    Tensor tanh(Tensor A, std::optional<Tensor> Y = std::nullopt);
    string printGraph() const;

    Graph getGraph() const;
};

} // namespace infini
#endif // GRAPH_BUILDER_H
