#pragma once
#include "core/graph.h"
#include "core/operator.h"

#include <infiniop/ops/clip.h>


namespace infini {
class ClipObj : public OperatorObj {
  public:
    /**
     * @brief Construct a new Clip object
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param min_val The minimum value tensor for clipping.
     * @param max_val The maximum value tensor for clipping.
     * @param output The output tensor.
     */
    ClipObj(GraphObj *graph, Tensor input, Tensor min_val, Tensor max_val, Tensor output);
    string toString() const override;
    ~ClipObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
};
} // namespace infini