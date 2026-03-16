#pragma once
#include "core/operator.h"

namespace infini {
class RMSNormObj : public OperatorObj {
  private:
    float eps;

  public:
    RMSNormObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output, float eps = 1e-6);
    std::optional<std::vector<ShapeExpr>> inferShape() override;
    std::vector<DataType> inferDataType() const override;
    std::string toString() const override;
    void createOpDesc() override;
    ~RMSNormObj() override;
    float getEps() const { return eps; }
};
} // namespace infini
