#pragma once
#include "core/operator.h"
#include <vector>

namespace infini {
class LayerNormObj : public OperatorObj {
  private:
    float eps;

  public:
    LayerNormObj(GraphObj *graph, Tensor input, Tensor weight, Tensor bias, Tensor output, float eps = 1e-5);

    std::optional<std::vector<ShapeExpr>> inferShape() override;
    std::vector<DataType> inferDataType() const override;

    std::string toString() const override;

    void createOpDesc() override;
    ~LayerNormObj() override;

    float getEps() const { return eps; }
};

} // namespace infini
