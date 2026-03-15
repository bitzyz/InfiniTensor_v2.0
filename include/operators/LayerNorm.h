#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/layer_norm.h>

namespace infini {

class LayerNormObj : public OperatorObj {
  private:
    int axis;
    float eps;

  public:
    // inputs = {input, weight, bias}, outputs = {y}
    // axis: dimension from which normalization begins (e.g. 1 means normalize
    //       over all dims >= 1; mean/std have shape input.shape[:axis]).
    LayerNormObj(GraphObj *graph, Tensor input, Tensor weight, Tensor bias,
                 Tensor output, int axis = 1, float eps = 1e-5f);
    string toString() const override;
    ~LayerNormObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    int getAxis() const;
    float getEps() const;
};

} // namespace infini
