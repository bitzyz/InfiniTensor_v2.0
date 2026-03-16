#pragma once
#include "core/operator.h"

namespace infini {
class TransposeObj : public OperatorObj {
  private:
    std::vector<int> perm;

  public:
    TransposeObj(GraphObj *graph, Tensor input, Tensor output, std::vector<int> perm);
    std::optional<std::vector<ShapeExpr>> inferShape() override;
    std::vector<DataType> inferDataType() const override;
    std::string toString() const override;
    void createOpDesc() override;
    ~TransposeObj() override;
    const std::vector<int>& getPerm() const { return perm; }
};
} // namespace infini
