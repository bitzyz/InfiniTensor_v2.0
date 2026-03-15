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
     * @param graph  Owning graph.
     * @param input  Input tensor.
     * @param output Output tensor (nullptr for auto-creation).
     * @param axis   Axis along which softmax is computed.
     */
    SoftmaxObj(GraphObj *graph, Tensor input, Tensor output, int axis);
    string toString() const override;
    ~SoftmaxObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    int getAxis() const;
};

} // namespace infini
