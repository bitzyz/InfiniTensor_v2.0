#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/gelu.h>
#include <infiniop/ops/relu.h>
#include <infiniop/ops/sigmoid.h>
#include <infiniop/ops/silu.h>
#include <infiniop/ops/softplus.h>
#include <infiniop/ops/tanh.h>

namespace infini {
class UnaryWiseObj : public OperatorObj {
  private:
    OpType type;

  public:
    /**
     * @brief Construct a new UnaryOpObj object
     *
     * @param type Operator type.
     * @param graph The computation graph that this operator belongs to.
     * @param input The first input tensor.
     * @param output The output tensor.
     */
    UnaryWiseObj(GraphObj *graph, OpType type, Tensor input, Tensor output);
    string toString() const override;
    ~UnaryWiseObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    OpType getUnaryWiseOpType() const;
};
} // namespace infini
