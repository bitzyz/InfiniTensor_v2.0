#pragma once
#include "core/operator.h"

namespace infini {
class LpNormObj : public OperatorObj {
  private:
    float p;
    std::vector<int> dims;
    bool keepdim;

  public:
    LpNormObj(GraphObj *graph, Tensor input, Tensor output, float p, std::vector<int> dims, bool keepdim);
    std::optional<std::vector<ShapeExpr>> inferShape() override;
    std::vector<DataType> inferDataType() const override;
    std::string toString() const override;
    void createOpDesc() override;
    ~LpNormObj() override;
    float getP() const { return p; }
    const std::vector<int>& getDims() const { return dims; }
    bool getKeepDim() const { return keepdim; }
};
} // namespace infini
