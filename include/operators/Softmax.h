#pragma once
#include "core/operator.h"

namespace infini {
class SoftmaxObj : public OperatorObj {
  private:
    int axis;

  public:
    SoftmaxObj(GraphObj *graph, Tensor input, Tensor output, int axis);
    std::optional<std::vector<ShapeExpr>> inferShape() override;
    std::vector<DataType> inferDataType() const override;
    std::string toString() const override;
    void createOpDesc() override;
    ~SoftmaxObj() override;
    int getAxis() const { return axis; }
};

class LogSoftmaxObj : public OperatorObj {
  private:
    int axis;

  public:
    LogSoftmaxObj(GraphObj *graph, Tensor input, Tensor output, int axis);
    std::optional<std::vector<ShapeExpr>> inferShape() override;
    std::vector<DataType> inferDataType() const override;
    std::string toString() const override;
    void createOpDesc() override;
    ~LogSoftmaxObj() override;
    int getAxis() const { return axis; }
};

} // namespace infini
