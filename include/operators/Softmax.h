#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/softmax.h>

namespace infini {
class SoftmaxObj : public OperatorObj {
  private:
    int axis;

  public:
    /**
     * @brief Construct a new Softmax object.
     * @param graph The computation graph that this operator belongs to.
     * @param x The input tensor.
     * @param axis The axis along which to compute softmax.
     * @param Y Y is the output of softmax.
     * Y should be an empty Ref.
     */
    SoftmaxObj(GraphObj *graph, Tensor x, int axis, Tensor Y);

    string toString() const override;
    ~SoftmaxObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
};
} // namespace infini
