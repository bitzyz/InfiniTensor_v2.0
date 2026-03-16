#pragma once
#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

#include "core/graph.h"
#include "core/op_type.h"
#include "operators/ElementWise.h"
#include "operators/Gemm.h"
#include "operators/Conv.h"
#include "operators/LayerNorm.h"
#include "operators/Unary.h"
#include "operators/Softmax.h"
#include "operators/RMSNorm.h"
#include "operators/LpNorm.h"
#include "operators/Transpose.h"

namespace infini {

class GraphBuilderObj {
  private:
    Ref<GraphObj> g;

  public:
    GraphBuilderObj(Runtime runtime);

    Tensor tensor(ShapeExpr dims, DataType dtype,
                  std::optional<StrideExpr> stride = std::nullopt);

    Tensor transpose(Tensor input, std::vector<int> perm, std::optional<Tensor> output = std::nullopt);

    Tensor gemm(Tensor A, Tensor B, Tensor C, float alpha = 1.0,
                float beta = 1.0, bool transA = false, bool transB = false,
                std::optional<Tensor> Y = std::nullopt);
    Tensor add(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor sub(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor mul(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor clip(Tensor input, Tensor min, Tensor max,
                std::optional<Tensor> output = std::nullopt);
    
    Tensor conv(Tensor input, Tensor weight, std::optional<Tensor> bias,
                std::vector<int> pads, std::vector<int> strides,
                std::vector<int> dilations, std::optional<Tensor> output = std::nullopt);

    Tensor layer_norm(Tensor input, Tensor weight, Tensor bias, float eps = 1e-5,
                      std::optional<Tensor> output = std::nullopt);

    Tensor relu(Tensor input, std::optional<Tensor> output = std::nullopt);
    Tensor sigmoid(Tensor input, std::optional<Tensor> output = std::nullopt);
    Tensor tanh(Tensor input, std::optional<Tensor> output = std::nullopt);
    Tensor gelu(Tensor input, std::optional<Tensor> output = std::nullopt);
    Tensor silu(Tensor input, std::optional<Tensor> output = std::nullopt);
    Tensor softplus(Tensor input, std::optional<Tensor> output = std::nullopt);
    
    Tensor softmax(Tensor input, int axis, std::optional<Tensor> output = std::nullopt);
    Tensor log_softmax(Tensor input, int axis, std::optional<Tensor> output = std::nullopt);
    
    Tensor rms_norm(Tensor input, Tensor weight, float eps = 1e-6, std::optional<Tensor> output = std::nullopt);
    
    Tensor lp_norm(Tensor input, float p, std::vector<int> dims, bool keepdim = false, std::optional<Tensor> output = std::nullopt);

    string printGraph() const;

    Graph getGraph() const;
};

} // namespace infini
#endif // GRAPH_BUILDER_H
