#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/lp_norm.h>

namespace infini {
class LpNormObj : public OperatorObj {
  private:
    int axis;
    int p;
    float eps;

  public:
    /**
     * @brief Construct a new LpNorm object.
     * @param graph The computation graph that this operator belongs to.
     * @param x The input tensor.
     * @param axis The axis along which to compute lp_norm.
     * @param p The p value for the p-norm calculation (1 for L1, 2 for L2).
     * @param eps A small value to avoid division by zero.
     * @param Y Y is the output of lp_norm.
     * Y should be an empty Ref.
     */
    LpNormObj(GraphObj *graph, Tensor x, int axis, int p, float eps, Tensor Y);

    string toString() const override;
    ~LpNormObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
};
} // namespace infini
