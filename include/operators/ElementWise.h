#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/add.h>
#include <infiniop/ops/clip.h>
#include <infiniop/ops/mul.h>
#include <infiniop/ops/sub.h>

namespace infini {
class ElementWiseObj : public OperatorObj {
  private:
    OpType type;

  public:
    /**
     * @brief Construct a new ElementWise object
     *
     * @param type Operator type.
     * @param graph The computation graph that this operator belongs to.
     * @param input0 The first input tensor.
     * @param input1 The second input tensor.
     * @param output The output tensor.
     */
    ElementWiseObj(GraphObj *graph, OpType type, Tensor input0, Tensor input1,
                   Tensor output);

    /**
     * @brief Construct a new ElementWise object for Clip
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param min The min tensor.
     * @param max The max tensor.
     * @param output The output tensor.
     */
    ElementWiseObj(GraphObj *graph, OpType type, Tensor input, Tensor min,
                   Tensor max, Tensor output);
    string toString() const override;
    ~ElementWiseObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    OpType getElemenwiseOpType() const;
};
} // namespace infini
