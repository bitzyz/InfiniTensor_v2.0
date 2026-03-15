#pragma once
#ifndef GRAPH_H
#define GRAPH_H

#include "core/operator.h"
#include <algorithm>
#include <infinirt.h>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace infini {
class GraphObj : public Object {
  protected:
    Runtime runtime;
    TensorVec tensors;
    TensorVec outputs;
    OpVec ops;
    bool cudaGraphEnabled = false;
    uint64_t mutationVersion = 0;
    uint64_t cudaGraphInvalidateCount = 0;
    uint64_t cudaGraphCompileCount = 0;
    uint64_t cudaGraphLaunchCount = 0;
    std::unordered_map<std::thread::id, uint64_t> compiledGraphVersions;
    std::unordered_map<std::thread::id, infinirtGraphExec_t> compiledGraphExecs;

  public:
    explicit GraphObj(Runtime runtime);
    ~GraphObj() override;
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
    const TensorVec &getOutputs() const;
    const OpVec &getOperators() const;
    Tensor getTensor(int) const;
    Runtime getRuntime() const;
    TensorVec getInputTensors() const;
    bool topo_sort();

    void shape_infer();
    void setOutputs(const TensorVec &newOutputs);
    void addOutput(const Tensor &tensor);
    void clearOutputs();
    void optimize();
    void replaceAllUses(const Tensor &oldTensor, const Tensor &newTensor);
    bool eraseOperator(const Operator &op);
    bool eraseTensorIfUnused(const Tensor &tensor);

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

    bool isCudaGraphEnabled() const;
    void setCudaGraphEnabled(bool enabled);
    uint64_t getMutationVersion() const;
    bool hasCompiledCudaGraphForCurrentThread() const;
    void markCudaGraphCompiledForCurrentThread();
    void markCudaGraphLaunched();
    infinirtGraphExec_t getCudaGraphExecForCurrentThread() const;
    void setCudaGraphExecForCurrentThread(infinirtGraphExec_t graphExec);
    void invalidateCudaGraph();
    uint64_t getCudaGraphInvalidateCount() const;
    uint64_t getCudaGraphCompileCount() const;
    uint64_t getCudaGraphLaunchCount() const;

  private:
    void addOperatorAndConnect(const Operator &op);
    void rebuildConnections();
};

} // namespace infini

#endif // GRAPH_H
