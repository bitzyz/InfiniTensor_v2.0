#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/logsoftmax.h>

namespace infini {
class LogSoftmaxObj : public OperatorObj {
  private:
    int dim;

  public:
    /**
     * @brief Construct a new LogSoftmax object.
     * @param graph The computation graph that this operator belongs to.
     * @param x The input tensor.
     * @param Y Y is the output of log_softmax.
     * Y should be an empty Ref.
     */
    LogSoftmaxObj(GraphObj *graph, Tensor x, int dim, Tensor Y);

    string toString() const override;
    ~LogSoftmaxObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
};
} // namespace infini
