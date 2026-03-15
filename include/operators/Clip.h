#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/clip.h>

namespace infini {
class ClipObj : public OperatorObj {

  public:
    /**
     * @brief Construct a new Gemm object.
     * @param graph The computation graph that this operator belongs to.
     * @param IN The input tensor.
     * @param Y Y is the output of clip.
     * Y should be an empty Ref.
     * @param MIN
     * @param MAX
     */
    ClipObj(GraphObj *graph, Tensor IN, Tensor Y, Tensor MIN, Tensor MAX);

    string toString() const override;
    ~ClipObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
};
} // namespace infini
