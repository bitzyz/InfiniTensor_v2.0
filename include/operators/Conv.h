#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/conv.h>

namespace infini {
class ConvObj : public OperatorObj {
  private:
    bool _has_bias;
    vector<size_t> _strides;
    vector<size_t> _dilations;
    vector<size_t> _paddings;
    int _n;

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
    ConvObj(GraphObj *graph, Tensor x, Tensor weight, Tensor bias,
            const vector<size_t> &strides, const vector<size_t> &dilations,
            const vector<size_t> &paddings, Tensor Y);

    string toString() const override;
    ~ConvObj() override;

    bool has_bias() const { return _has_bias; }
    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
};
} // namespace infini
