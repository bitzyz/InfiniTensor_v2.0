#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/gelu.h>
#include <infiniop/ops/relu.h>
#include <infiniop/ops/sigmoid.h>
#include <infiniop/ops/silu.h>
#include <infiniop/ops/softplus.h>
#include <infiniop/ops/tanh.h>

namespace infini {

/**
 * @brief Elementwise unary activation operators: Relu, Sigmoid, Silu, Gelu,
 *        Softplus, Tanh.
 *
 * All share the same shape/dtype inference (output mirrors input) and the same
 * InfiniOp API shape: infiniopCreate{Op}Descriptor(handle, &desc, y, x).
 */
class UnaryObj : public OperatorObj {
  public:
    /**
     * @param graph  Owning graph.
     * @param type   One of OpType::{Relu,Sigmoid,Silu,Gelu,Softplus,Tanh}.
     * @param input  Input tensor.
     * @param output Output tensor (nullptr for auto-creation).
     */
    UnaryObj(GraphObj *graph, OpType type, Tensor input, Tensor output);
    string toString() const override;
    ~UnaryObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    OpType getUnaryOpType() const;
};

} // namespace infini
