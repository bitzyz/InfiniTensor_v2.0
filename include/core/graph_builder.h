#pragma once
#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

#include "core/graph.h"
#include "core/op_type.h"
#include "operators/Clip.h"
#include "operators/Conv.h"
#include "operators/ElementWise.h"
#include "operators/Gemm.h"
#include "operators/LPNorm.h"
#include "operators/LayerNorm.h"
#include "operators/LogSoftmax.h"
#include "operators/RMSNorm.h"
#include "operators/Softmax.h"
#include "operators/Unary.h"

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
    void set_outputs(const TensorVec &outputs);
    void optimize();

    // Convolution
    Tensor conv(Tensor x, Tensor w, Tensor bias, vector<int64_t> pads,
                vector<int64_t> strides, vector<int64_t> dilations,
                std::optional<Tensor> y = std::nullopt);

    // Clip
    Tensor clip(Tensor input, Tensor min_val, Tensor max_val,
                std::optional<Tensor> output = std::nullopt);

    // Softmax / LogSoftmax
    Tensor softmax(Tensor input, int axis,
                   std::optional<Tensor> output = std::nullopt);
    Tensor log_softmax(Tensor input,
                       std::optional<Tensor> output = std::nullopt);

    // Normalization
    Tensor layer_norm(Tensor input, Tensor weight, Tensor bias,
                      std::optional<Tensor> output = std::nullopt, int axis = 1,
                      float eps = 1e-5f);
    Tensor rms_norm(Tensor input, Tensor weight,
                    std::optional<Tensor> output = std::nullopt,
                    float epsilon = 1e-6f);
    Tensor lp_norm(Tensor input, int axis, int p,
                   std::optional<Tensor> output = std::nullopt,
                   float eps = 1e-12f);

    // Unary activations
    Tensor relu(Tensor input, std::optional<Tensor> output = std::nullopt);
    Tensor sigmoid(Tensor input, std::optional<Tensor> output = std::nullopt);
    Tensor silu(Tensor input, std::optional<Tensor> output = std::nullopt);
    Tensor gelu(Tensor input, std::optional<Tensor> output = std::nullopt);
    Tensor softplus(Tensor input, std::optional<Tensor> output = std::nullopt);
    Tensor tanh(Tensor input, std::optional<Tensor> output = std::nullopt);

    string printGraph() const;

    Graph getGraph() const;
};

} // namespace infini
#endif // GRAPH_BUILDER_H
