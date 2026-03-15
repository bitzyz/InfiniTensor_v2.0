#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/layer_norm.h>

namespace infini {
class LayerNormObj : public OperatorObj {
  private:
    bool _has_bias;
    float _eps;

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
    LayerNormObj(GraphObj *graph, Tensor x, Tensor weight, Tensor bias,
                 float eps, Tensor Y, Tensor Norm, Tensor Std);

    string toString() const override;
    ~LayerNormObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
    bool has_bias() { return _has_bias; }
};
} // namespace infini
