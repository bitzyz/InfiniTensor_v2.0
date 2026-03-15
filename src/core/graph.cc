#include "core/graph.h"
#include "core/graph_optimizer.h"

namespace infini {

namespace {

void destroyGraphExecNoThrow(infinirtGraphExec_t graphExec) {
    if (graphExec == nullptr) {
        return;
    }
    auto err = infinirtGraphExecDestroy(graphExec);
    if (err != INFINI_STATUS_SUCCESS) {
        std::cerr << "[WARN] Failed to destroy cuda graph exec, error=" << err
                  << std::endl;
    }
}

} // namespace

GraphObj::GraphObj(Runtime runtime) : runtime(runtime) {}

GraphObj::~GraphObj() {
    for (auto &kv : compiledGraphExecs) {
        destroyGraphExecNoThrow(kv.second);
    }
}

std::string GraphObj::toString() const {
    std::ostringstream oss;
    oss << "=== Graph ===\n";

    oss << "[Tensors]\n";
    for (const auto &tensor : tensors) {
        oss << "  " << tensor << "\n";
    }

    oss << "[Operators]\n";
    for (const auto &op : ops) {
        oss << "  OP " << op->getGuid() << "\n";

        oss << "    Preds: ";
        for (auto &o : op->getPredecessors())
            oss << o->getGuid() << " ";
        oss << "\n";

        oss << "    Succs: ";
        for (auto &o : op->getSuccessors())
            oss << o->getGuid() << " ";
        oss << "\n";

        oss << "    Detail: " << op << "\n";
    }

    oss << "[Outputs]\n";
    for (const auto &tensor : outputs) {
        oss << "  Tensor " << tensor->getGuid() << "\n";
    }
    return oss.str();
}

Runtime GraphObj::getRuntime() const { return runtime; }

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
    auto tensor = tensors.emplace_back(make_ref<TensorObj>(dim, dtype));
    invalidateCudaGraph();
    return tensor;
}

Tensor GraphObj::addTensor(Shape dim, Stride stride, DataType dtype) {
    auto tensor = tensors.emplace_back(make_ref<TensorObj>(dim, stride, dtype));
    invalidateCudaGraph();
    return tensor;
}

Tensor GraphObj::addTensor(Shape dim, StrideExpr stride, DataType dtype) {
    auto tensor = tensors.emplace_back(make_ref<TensorObj>(dim, stride, dtype));
    invalidateCudaGraph();
    return tensor;
}

Tensor GraphObj::addTensor(ShapeExpr dim, DataType dtype) {
    auto tensor = tensors.emplace_back(make_ref<TensorObj>(dim, dtype));
    invalidateCudaGraph();
    return tensor;
}

Tensor GraphObj::addTensor(ShapeExpr dim, Stride stride, DataType dtype) {
    auto tensor = tensors.emplace_back(make_ref<TensorObj>(dim, stride, dtype));
    invalidateCudaGraph();
    return tensor;
}

Tensor GraphObj::addTensor(ShapeExpr dim, StrideExpr stride, DataType dtype) {
    auto tensor = tensors.emplace_back(make_ref<TensorObj>(dim, stride, dtype));
    invalidateCudaGraph();
    return tensor;
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
    tensors.emplace_back(tensor);
    invalidateCudaGraph();
    return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
    for (auto &t : tensors)
        addTensor(t);
    return tensors;
}

void GraphObj::removeOperator(Operator op) {
    auto it = std::find(ops.begin(), ops.end(), op);
    if (it != ops.end()) {
        ops.erase(it);
        invalidateCudaGraph();
    }
}

void GraphObj::removeTensor(Tensor tensor) {
    auto it = std::find(tensors.begin(), tensors.end(), tensor);
    if (it != tensors.end()) {
        tensors.erase(it);
        invalidateCudaGraph();
    }
}

const TensorVec &GraphObj::getTensors() const { return tensors; }

const TensorVec &GraphObj::getOutputs() const { return outputs; }

const OpVec &GraphObj::getOperators() const { return ops; }

Tensor GraphObj::getTensor(int fuid) const {
    for (auto tensor : tensors) {
        if (tensor->getFuid() == fuid) {
            return tensor;
        }
    }
    return nullptr;
}

TensorVec GraphObj::getInputTensors() const {
    TensorVec inputs;
    for (const auto &tensor : tensors) {
        if (tensor->isGraphInput()) {
            inputs.emplace_back(tensor);
        }
    }
    return inputs;
}

bool GraphObj::topo_sort() {
    std::unordered_map<OperatorObj *, int> indegree;
    for (auto &op : ops) {
        indegree[op.get()] = 0;
    }
    for (auto &op : ops) {
        for (auto &input : op->getInputs()) {
            if (auto src = input->getSource()) {
                indegree[op.get()]++;
            }
        }
    }

    std::queue<Operator> q;
    for (auto &op : ops) {
        if (indegree[op.get()] == 0) {
            q.push(op);
        }
    }

    std::vector<Operator> sorted;
    while (!q.empty()) {
        auto op = q.front();
        q.pop();
        sorted.push_back(op);

        for (auto &succ : op->getSuccessors()) {
            if (--indegree[succ.get()] == 0) {
                q.push(succ);
            }
        }
    }

    if (sorted.size() != ops.size()) {
        // Cycle detected, topological sort failed
        return false;
    }

    if (sorted != ops) {
        invalidateCudaGraph();
    }
    ops = std::move(sorted);
    return true;
}

void GraphObj::shape_infer() {
    for (auto &op : ops) {
        auto ans = op->inferShape();
        IT_ASSERT(ans.has_value());
        auto oldOutputs = op->getOutputs();
        IT_ASSERT(ans.value().size() == oldOutputs.size());
        // replace the old outputshape and size with new one
        for (int i = 0; i < (int)ans.value().size(); ++i) {
            auto newShape = ans.value()[i];
            auto oldShape = oldOutputs[i]->getShape();
            auto fuid = oldOutputs[i]->getFuid();
            if (newShape != oldShape) {
                auto tensor = this->getTensor(fuid);
                tensor->setShape(newShape);
                invalidateCudaGraph();
            }
        }
    }
}

void GraphObj::setOutputs(const TensorVec &newOutputs) {
    if (outputs == newOutputs) {
        return;
    }
    outputs.clear();
    for (const auto &tensor : newOutputs) {
        IT_ASSERT(tensor != nullptr, "Graph output tensor cannot be null");
        IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end(),
                  "Graph output must belong to graph");
        if (std::find(outputs.begin(), outputs.end(), tensor) ==
            outputs.end()) {
            outputs.emplace_back(tensor);
        }
    }
    invalidateCudaGraph();
}

void GraphObj::addOutput(const Tensor &tensor) {
    IT_ASSERT(tensor != nullptr, "Graph output tensor cannot be null");
    IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                  tensors.end(),
              "Graph output must belong to graph");
    if (std::find(outputs.begin(), outputs.end(), tensor) == outputs.end()) {
        outputs.emplace_back(tensor);
        invalidateCudaGraph();
    }
}

void GraphObj::clearOutputs() {
    if (!outputs.empty()) {
        outputs.clear();
        invalidateCudaGraph();
    }
}

void GraphObj::optimize() {
    GraphOptimizer().run(*this);
    invalidateCudaGraph();
}

bool GraphObj::checkBeforRun() const {
    for (auto tensor : tensors) {
        auto shape = tensor->getShape();
        auto stride = tensor->getStride();
        IT_ASSERT(shape->isConcrete(),
                  "Shape of tensor " + tensor->toString() + " is not concrete");
        IT_ASSERT(stride->isConcrete(), "Stride of tensor " +
                                            tensor->toString() +
                                            " is not concrete");
    }
    return true;
}

void GraphObj::addOperatorAndConnect(const Operator &op) {
    ops.push_back(op);
    for (auto &input : op->getInputs()) {
        if (input) {
            input->addTarget(op);
            if (auto pred = input->getSource()) {
                pred->addSuccessors(op);
                op->addPredecessors(pred);
            }
        }
    }
    for (auto &output : op->getOutputs()) {
        if (output) {
            output->setSource(op);
            for (auto &succ : output->getTargets()) {
                succ->addPredecessors(op);
                op->addSuccessors(succ);
            }
        }
    }
    invalidateCudaGraph();
}

void GraphObj::rebuildConnections() {
    for (auto &tensor : tensors) {
        tensor->clearTargets();
        tensor->clearSource();
    }
    for (auto &op : ops) {
        op->clearConnections();
    }

    for (auto &op : ops) {
        for (auto &input : op->getInputs()) {
            if (!input) {
                continue;
            }
            input->addTarget(op);
            if (auto pred = input->getSource()) {
                pred->addSuccessors(op);
                op->addPredecessors(pred);
            }
        }
        for (auto &output : op->getOutputs()) {
            if (output) {
                output->setSource(op);
            }
        }
    }
}

void GraphObj::replaceAllUses(const Tensor &oldTensor,
                              const Tensor &newTensor) {
    IT_ASSERT(oldTensor != nullptr && newTensor != nullptr,
              "replaceAllUses expects non-null tensors");
    IT_ASSERT(std::find(tensors.begin(), tensors.end(), oldTensor) !=
                  tensors.end(),
              "Old tensor must belong to graph");
    IT_ASSERT(std::find(tensors.begin(), tensors.end(), newTensor) !=
                  tensors.end(),
              "New tensor must belong to graph");

    for (auto &op : ops) {
        op->replaceInput(oldTensor, newTensor);
    }
    for (auto &output : outputs) {
        if (output == oldTensor) {
            output = newTensor;
        }
    }
    rebuildConnections();
    invalidateCudaGraph();
}

bool GraphObj::eraseOperator(const Operator &op) {
    auto it = std::find(ops.begin(), ops.end(), op);
    if (it == ops.end()) {
        return false;
    }
    ops.erase(it);
    rebuildConnections();
    invalidateCudaGraph();
    return true;
}

bool GraphObj::eraseTensorIfUnused(const Tensor &tensor) {
    auto it = std::find(tensors.begin(), tensors.end(), tensor);
    if (it == tensors.end()) {
        return false;
    }
    if (std::find(outputs.begin(), outputs.end(), tensor) != outputs.end() ||
        tensor->getSource() != nullptr || !tensor->getTargets().empty()) {
        return false;
    }
    tensors.erase(it);
    invalidateCudaGraph();
    return true;
}

bool GraphObj::isCudaGraphEnabled() const { return cudaGraphEnabled; }

void GraphObj::setCudaGraphEnabled(bool enabled) {
    if (cudaGraphEnabled != enabled) {
        cudaGraphEnabled = enabled;
        invalidateCudaGraph();
    }
}

uint64_t GraphObj::getMutationVersion() const { return mutationVersion; }

bool GraphObj::hasCompiledCudaGraphForCurrentThread() const {
    const auto tid = std::this_thread::get_id();
    auto it = compiledGraphVersions.find(tid);
    return it != compiledGraphVersions.end() && it->second == mutationVersion;
}

void GraphObj::markCudaGraphCompiledForCurrentThread() {
    compiledGraphVersions[std::this_thread::get_id()] = mutationVersion;
    ++cudaGraphCompileCount;
}

void GraphObj::markCudaGraphLaunched() { ++cudaGraphLaunchCount; }

infinirtGraphExec_t GraphObj::getCudaGraphExecForCurrentThread() const {
    auto it = compiledGraphExecs.find(std::this_thread::get_id());
    return it == compiledGraphExecs.end() ? nullptr : it->second;
}

void GraphObj::setCudaGraphExecForCurrentThread(infinirtGraphExec_t graphExec) {
    auto tid = std::this_thread::get_id();
    auto it = compiledGraphExecs.find(tid);
    if (it != compiledGraphExecs.end()) {
        destroyGraphExecNoThrow(it->second);
        it->second = graphExec;
    } else {
        compiledGraphExecs.emplace(tid, graphExec);
    }
}

void GraphObj::invalidateCudaGraph() {
    ++mutationVersion;
    ++cudaGraphInvalidateCount;
    for (auto &kv : compiledGraphExecs) {
        destroyGraphExecNoThrow(kv.second);
    }
    compiledGraphExecs.clear();
    compiledGraphVersions.clear();
}

uint64_t GraphObj::getCudaGraphInvalidateCount() const {
    return cudaGraphInvalidateCount;
}

uint64_t GraphObj::getCudaGraphCompileCount() const {
    return cudaGraphCompileCount;
}

uint64_t GraphObj::getCudaGraphLaunchCount() const {
    return cudaGraphLaunchCount;
}

bool GraphObj::checkValid() const {
    // Build fast lookup sets
    std::unordered_set<Tensor> tensorSet(tensors.begin(), tensors.end());
    std::unordered_set<Operator> opSet(ops.begin(), ops.end());

    // 1. Check all Tensors
    for (auto tensor : tensors) {
        // Must have source or targets
        bool isGraphOutput =
            std::find(outputs.begin(), outputs.end(), tensor) != outputs.end();
        IT_ASSERT(!(tensor->getTargets().empty() &&
                    tensor->getSource() == nullptr && !isGraphOutput),
                  "Invalid tensor: " + tensor->toString() +
                      " has no source, no targets, and is not a graph output");

        // Check if all target ops are in Graph
        for (auto op : tensor->getTargets()) {
            IT_ASSERT(opSet.count(op),
                      "Tensor " + tensor->toString() +
                          " has target op not in graph: " + op->toString());
        }

        // Check if source op is in Graph
        if (auto src = tensor->getSource()) {
            IT_ASSERT(opSet.count(src),
                      "Tensor " + tensor->toString() +
                          " has source op not in graph: " + src->toString());
        }
    }

    // 2. Check all Operators
    for (auto op : ops) {
        // Input tensor must exist
        for (auto tensor : op->getInputs()) {
            IT_ASSERT(
                tensorSet.count(tensor),
                "Op " + op->toString() +
                    " has input tensor not in graph: " + tensor->toString());
        }
        // Output tensor must exist
        for (auto tensor : op->getOutputs()) {
            IT_ASSERT(
                tensorSet.count(tensor),
                "Op " + op->toString() +
                    " has output tensor not in graph: " + tensor->toString());
        }
        // Predecessors/Successors must exist
        for (auto pre : op->getPredecessors()) {
            IT_ASSERT(opSet.count(pre),
                      "Op " + op->toString() +
                          " has predecessor not in graph: " + pre->toString());
        }
        for (auto suc : op->getSuccessors()) {
            IT_ASSERT(opSet.count(suc),
                      "Op " + op->toString() +
                          " has successor not in graph: " + suc->toString());
        }
    }

    // 3. Check uniqueness of Tensor FUIDs
    std::unordered_set<UidBaseType> fuids;
    for (auto tensor : tensors) {
        IT_ASSERT(fuids.insert(tensor->getFuid()).second,
                  "Duplicate tensor fuid: " +
                      std::to_string(tensor->getFuid()));
    }

    // 4. Check bidirectional consistency
    for (auto tensor : tensors) {
        for (auto targetOp : tensor->getTargets()) {
            auto &inputs = targetOp->getInputs();
            IT_ASSERT(std::find(inputs.begin(), inputs.end(), tensor) !=
                          inputs.end(),
                      "Mismatch: tensor " + tensor->toString() +
                          " lists target op " + targetOp->toString() +
                          " but that op does not have this tensor as input");
        }
        if (auto srcOp = tensor->getSource()) {
            auto &outputs = srcOp->getOutputs();
            IT_ASSERT(std::find(outputs.begin(), outputs.end(), tensor) !=
                          outputs.end(),
                      "Mismatch: tensor " + tensor->toString() +
                          " has source op " + srcOp->toString() +
                          " but that op does not have this tensor as output");
        }
    }

    for (auto tensor : outputs) {
        IT_ASSERT(tensorSet.count(tensor),
                  "Graph output tensor not in graph: " + tensor->toString());
    }

    return true;
}

} // namespace infini
