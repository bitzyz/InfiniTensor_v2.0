#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/rms_norm.h>

namespace infini {
class RmsNormObj : public OperatorObj {
  private:
    float _eps;

  public:
    /**
     * @brief Construct a new RmsNorm object.
     * @param graph The computation graph that this operator belongs to.
     * @param x The input tensor.
     * @param weight The weight tensor.
     * @param eps A small value to avoid division by zero.
     * @param Y Y is the output of rms_norm.
     * Y should be an empty Ref.
     */
    RmsNormObj(GraphObj *graph, Tensor x, Tensor weight, float eps, Tensor Y);

    string toString() const override;
    ~RmsNormObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
};
} // namespace infini
