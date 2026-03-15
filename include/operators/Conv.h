#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/conv.h>

namespace infini {

class ConvObj : public OperatorObj {
  private:
    // All vectors are of length n_spatial_dims = rank(x) - 2.
    // pads: symmetric padding per spatial dim.
    // strides, dilations: per spatial dim.
    vector<int64_t> pads;
    vector<int64_t> strides;
    vector<int64_t> dilations;

  public:
    // inputs = {x, w} or {x, w, bias}; outputs = {y}.
    // Bias is optional: pass nullptr to omit.
    ConvObj(GraphObj *graph, Tensor x, Tensor w, Tensor bias, Tensor y,
            vector<int64_t> pads, vector<int64_t> strides,
            vector<int64_t> dilations);
    string toString() const override;
    ~ConvObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    const vector<int64_t> &getPads() const;
    const vector<int64_t> &getStrides() const;
    const vector<int64_t> &getDilations() const;
    bool hasBias() const;
};

} // namespace infini
