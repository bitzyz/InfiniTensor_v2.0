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
     * @param min_val The min-clamp tensor (scalar-shaped).
     * @param max_val The max-clamp tensor (scalar-shaped).
     * @param output The output tensor (may be nullptr for auto-creation).
     */
    ClipObj(GraphObj *graph, Tensor input, Tensor min_val, Tensor max_val,
            Tensor output);
    string toString() const override;
    ~ClipObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
};
} // namespace infini