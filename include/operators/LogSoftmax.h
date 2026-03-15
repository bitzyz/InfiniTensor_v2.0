#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/logsoftmax.h>

namespace infini {

/**
 * @brief LogSoftmax operator.  The InfiniOp API applies reduction over all
 *        elements (no axis parameter); shape and dtype pass through unchanged.
 */
class LogSoftmaxObj : public OperatorObj {
  public:
    /**
     * @param graph  Owning graph.
     * @param input  Input tensor.
     * @param output Output tensor (nullptr for auto-creation).
     */
    LogSoftmaxObj(GraphObj *graph, Tensor input, Tensor output);
    string toString() const override;
    ~LogSoftmaxObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
};

} // namespace infini
