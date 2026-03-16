#pragma once
#include "core/operator.h"

namespace infini {
class UnaryObj : public OperatorObj {
  public:
    UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output);
    std::optional<std::vector<ShapeExpr>> inferShape() override;
    std::vector<DataType> inferDataType() const override;
    std::string toString() const override;
    void createOpDesc() override;
    ~UnaryObj() override;
};

class ReluObj : public UnaryObj {
  public:
    ReluObj(GraphObj *graph, Tensor input, Tensor output)
        : UnaryObj(OpType::Relu, graph, input, output) {}
};

class SigmoidObj : public UnaryObj {
  public:
    SigmoidObj(GraphObj *graph, Tensor input, Tensor output)
        : UnaryObj(OpType::Sigmoid, graph, input, output) {}
};

class TanhObj : public UnaryObj {
  public:
    TanhObj(GraphObj *graph, Tensor input, Tensor output)
        : UnaryObj(OpType::Tanh, graph, input, output) {}
};

class GeluObj : public UnaryObj {
  public:
    GeluObj(GraphObj *graph, Tensor input, Tensor output)
        : UnaryObj(OpType::Gelu, graph, input, output) {}
};

class SiluObj : public UnaryObj {
  public:
    SiluObj(GraphObj *graph, Tensor input, Tensor output)
        : UnaryObj(OpType::Silu, graph, input, output) {}
};

class SoftplusObj : public UnaryObj {
  public:
    SoftplusObj(GraphObj *graph, Tensor input, Tensor output)
        : UnaryObj(OpType::Softplus, graph, input, output) {}
};

} // namespace infini
