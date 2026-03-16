#pragma once
#include "core/operator.h"
#include <vector>

namespace infini {
class ConvObj : public OperatorObj {
  private:
    std::vector<int> pads;
    std::vector<int> strides;
    std::vector<int> dilations;

  public:
    ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
            std::vector<int> pads, std::vector<int> strides,
            std::vector<int> dilations, Tensor bias = nullptr);

    std::optional<std::vector<ShapeExpr>> inferShape() override;
    std::vector<DataType> inferDataType() const override;

    std::string toString() const override;

    void createOpDesc() override;
    ~ConvObj() override;

    const std::vector<int>& getPads() const { return pads; }
    const std::vector<int>& getStrides() const { return strides; }
    const std::vector<int>& getDilations() const { return dilations; }
};

} // namespace infini
