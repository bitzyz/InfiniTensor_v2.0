#pragma once
#ifndef OPERATOR_H
#define OPERATOR_H

#include "core/op_type.h"
#include "core/tensor.h"

namespace infini {

class OperatorObj : public Object {
    friend class GraphObj;

  protected:
    OpType type;
    TensorVec inputs;
    TensorVec outputs;
    vector<WRef<OperatorObj>> predecessors;
    vector<WRef<OperatorObj>> successors;
    void *infiniOpDesc = nullptr;

  public:
    OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs);
    const TensorVec &getInputs() const;
    const TensorVec &getOutputs() const;
    Tensor getInput(size_t idx) const;
    Tensor getOutput(size_t idx) const;
    OpType getOpType() const;
    OpVec getPredecessors() const;
    OpVec getSuccessors() const;
    DataType getInDType(size_t idx) const;
    DataType getOutDType(size_t idx) const;
    ElementType getNumInputs() const;
    ElementType getNumOutputs() const;
    virtual void createOpDesc() = 0;
    void *getInfiniOpDesc() const;

  protected:
    virtual optional<vector<ShapeExpr>> inferShape() = 0;
    virtual vector<DataType> inferDataType() const = 0;
    bool checkValid(GraphObj *graph);

  private:
    void addPredecessors(const Operator &op);
    void addSuccessors(const Operator &op);
    void removePredecessors(const Operator &op);
    void removeSuccessors(const Operator &op);
    void replaceInput(Tensor t1, Tensor t2);
};

} // namespace infini

#endif // OPERATOR_H
