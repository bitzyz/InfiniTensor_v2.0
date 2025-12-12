#pragma once
#ifndef GRAPH_H
#define GRAPH_H

#include "core/operator.h"
#include <algorithm>
#include <numeric>

namespace infini {
class GraphObj : public Object {
  protected:
    Runtime runtime;
    TensorVec tensors;
    OpVec ops;

  public:
    explicit GraphObj(Runtime runtime);
    string toString() const override;

    Tensor addTensor(Shape dim, DataType dtype);
    Tensor addTensor(Shape dim, Stride stride, DataType dtype);
    Tensor addTensor(Shape dim, StrideExpr stride, DataType dtype);
    Tensor addTensor(ShapeExpr dim, DataType dtype);
    Tensor addTensor(ShapeExpr dim, StrideExpr stride, DataType dtype);
    Tensor addTensor(ShapeExpr dim, Stride stride, DataType dtype);
    Tensor addTensor(const Tensor &tensor);
    TensorVec addTensor(const TensorVec &tensors);
    void removeOperator(Operator op);
    void removeTensor(Tensor tensor);
    const TensorVec &getTensors() const;
    const OpVec &getOperators() const;
    Tensor getTensor(int) const;
    Runtime getRuntime() const;
    bool topo_sort();

    void shape_infer();

    template <typename T, typename... Args> Ref<T> addOp(Args &&...args) {
        Ref<T> op = infini::make_ref<T>(this, std::forward<Args>(args)...);
        addOperatorAndConnect(op);
        return op;
    }

    template <typename T, typename... Args>
    Ref<T> addOpWithOutputs(Args &&...args) {
        Ref<T> op = infini::make_ref<T>(nullptr, std::forward<Args>(args)...);
        addOperatorAndConnect(op);
        return op;
    }

    bool checkValid() const;
    bool checkBeforRun() const;

  private:
    void addOperatorAndConnect(const Operator &op);
};

} // namespace infini

#endif // GRAPH_H
