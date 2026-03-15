#include "core/graph_optimizer.h"

#include "operators/Clip.h"
#include "operators/Softmax.h"
#include "operators/Unary.h"

#include "utils/utils.h"
#include <cmath>
#include <memory>

namespace infini {
namespace {

bool tensorHasConcreteF32Data(const Tensor &tensor) {
    return tensor != nullptr && tensor->hasData() &&
           tensor->getSource() == nullptr &&
           tensor->getDevice() == INFINI_DEVICE_CPU &&
           tensor->getDataType() == DataType(INFINI_DTYPE_F32) &&
           tensor->getShape()->isConcrete() &&
           tensor->getStride()->isConcrete();
}

bool isScalarConstantValue(const Tensor &tensor, float expected) {
    if (!tensorHasConcreteF32Data(tensor) || tensor->getElement() != 1) {
        return false;
    }
    return std::fabs(tensor->getRawDataPtr<float *>()[0] - expected) < 1e-6f;
}

optional<float> getScalarConstantValue(const Tensor &tensor) {
    if (!tensorHasConcreteF32Data(tensor) || tensor->getElement() != 1) {
        return {};
    }
    return tensor->getRawDataPtr<float *>()[0];
}

bool isIdentityUnaryOp(OpType opType) { return opType == OpType::Relu; }

class TopologicalSortPass final : public GraphPass {
  public:
    string name() const override { return "TopologicalSortPass"; }
    bool run(GraphObj &graph) override { return graph.topo_sort(); }
};

class ShapeInferPass final : public GraphPass {
  public:
    string name() const override { return "ShapeInferPass"; }
    bool run(GraphObj &graph) override {
        graph.shape_infer();
        return true;
    }
};

class ConstantFoldPass final : public GraphPass {
  public:
    string name() const override { return "ConstantFoldPass"; }

    bool run(GraphObj &graph) override {
        bool changed = false;
        auto ops = graph.getOperators();
        for (const auto &op : ops) {
            if (op->getNumOutputs() != 1) {
                continue;
            }
            auto output = op->getOutput(0);
            if (!output || !output->getShape()->isConcrete() ||
                !output->getStride()->isConcrete() ||
                output->getDataType() != DataType(INFINI_DTYPE_F32)) {
                continue;
            }

            bool allInputsConstant = true;
            for (const auto &input : op->getInputs()) {
                if (!tensorHasConcreteF32Data(input)) {
                    allInputsConstant = false;
                    break;
                }
            }
            if (!allInputsConstant) {
                continue;
            }

            const auto outShape = output->getShape()->getConstantValue();
            const auto numel = output->getElement();
            auto *buffer = new float[numel];

            if (op->getOpType() == OpType::Add ||
                op->getOpType() == OpType::Sub ||
                op->getOpType() == OpType::Mul ||
                op->getOpType() == OpType::Div) {
                auto lhs = op->getInput(0);
                auto rhs = op->getInput(1);
                const auto lhsStride =
                    broadcastStride(lhs->getShape(), lhs->getStride(),
                                    output->getShape())
                        ->getConstantValue();
                const auto rhsStride =
                    broadcastStride(rhs->getShape(), rhs->getStride(),
                                    output->getShape())
                        ->getConstantValue();
                const auto *lhsPtr = lhs->getRawDataPtr<float *>();
                const auto *rhsPtr = rhs->getRawDataPtr<float *>();

                for (size_t i = 0; i < static_cast<size_t>(numel); ++i) {
                    auto lhsOffset =
                        calculateLinearOffset(i, outShape, lhsStride);
                    auto rhsOffset =
                        calculateLinearOffset(i, outShape, rhsStride);
                    switch (op->getOpType().underlying()) {
                    case OpType::Add:
                        buffer[i] = lhsPtr[lhsOffset] + rhsPtr[rhsOffset];
                        break;
                    case OpType::Sub:
                        buffer[i] = lhsPtr[lhsOffset] - rhsPtr[rhsOffset];
                        break;
                    case OpType::Mul:
                        buffer[i] = lhsPtr[lhsOffset] * rhsPtr[rhsOffset];
                        break;
                    case OpType::Div:
                        buffer[i] = lhsPtr[lhsOffset] / rhsPtr[rhsOffset];
                        break;
                    default:
                        IT_ASSERT(false, "Unexpected elementwise op");
                    }
                }
            } else if (op->getOpType() == OpType::Clip) {
                auto input = op->getInput(0);
                auto minVal = op->getInput(1);
                auto maxVal = op->getInput(2);

                const auto inStride =
                    broadcastStride(input->getShape(), input->getStride(),
                                    output->getShape())
                        ->getConstantValue();
                const auto minStride =
                    broadcastStride(minVal->getShape(), minVal->getStride(),
                                    output->getShape())
                        ->getConstantValue();
                const auto maxStride =
                    broadcastStride(maxVal->getShape(), maxVal->getStride(),
                                    output->getShape())
                        ->getConstantValue();
                const auto *inPtr = input->getRawDataPtr<float *>();
                const auto *minPtr = minVal->getRawDataPtr<float *>();
                const auto *maxPtr = maxVal->getRawDataPtr<float *>();

                for (size_t i = 0; i < static_cast<size_t>(numel); ++i) {
                    auto inOffset =
                        calculateLinearOffset(i, outShape, inStride);
                    auto minOffset =
                        calculateLinearOffset(i, outShape, minStride);
                    auto maxOffset =
                        calculateLinearOffset(i, outShape, maxStride);
                    auto clipped = std::max(inPtr[inOffset], minPtr[minOffset]);
                    buffer[i] = std::min(clipped, maxPtr[maxOffset]);
                }
            } else if (isIdentityUnaryOp(op->getOpType())) {
                auto input = op->getInput(0);
                const auto inStride =
                    broadcastStride(input->getShape(), input->getStride(),
                                    output->getShape())
                        ->getConstantValue();
                const auto *inPtr = input->getRawDataPtr<float *>();

                for (size_t i = 0; i < static_cast<size_t>(numel); ++i) {
                    auto inOffset =
                        calculateLinearOffset(i, outShape, inStride);
                    buffer[i] = std::max(inPtr[inOffset], 0.0f);
                }
            } else if (op->getOpType() == OpType::Sigmoid ||
                       op->getOpType() == OpType::Silu ||
                       op->getOpType() == OpType::Gelu ||
                       op->getOpType() == OpType::Softplus ||
                       op->getOpType() == OpType::Tanh) {
                auto input = op->getInput(0);
                const auto inStride =
                    broadcastStride(input->getShape(), input->getStride(),
                                    output->getShape())
                        ->getConstantValue();
                const auto *inPtr = input->getRawDataPtr<float *>();

                for (size_t i = 0; i < static_cast<size_t>(numel); ++i) {
                    auto inOffset =
                        calculateLinearOffset(i, outShape, inStride);
                    const float x = inPtr[inOffset];
                    switch (op->getOpType().underlying()) {
                    case OpType::Sigmoid:
                        buffer[i] = 1.0f / (1.0f + std::exp(-x));
                        break;
                    case OpType::Silu:
                        buffer[i] = x / (1.0f + std::exp(-x));
                        break;
                    case OpType::Gelu:
                        buffer[i] =
                            0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
                        break;
                    case OpType::Softplus:
                        buffer[i] = std::log1p(std::exp(x));
                        break;
                    case OpType::Tanh:
                        buffer[i] = std::tanh(x);
                        break;
                    default:
                        IT_ASSERT(false, "Unexpected unary op");
                    }
                }
            } else {
                delete[] buffer;
                continue;
            }

            output->setData(buffer);
            graph.eraseOperator(op);
            changed = true;
        }
        return changed;
    }
};

class IdentityEliminationPass final : public GraphPass {
  public:
    string name() const override { return "IdentityEliminationPass"; }

    bool run(GraphObj &graph) override {
        bool changed = false;
        auto ops = graph.getOperators();
        for (const auto &op : ops) {
            Tensor replacement = nullptr;
            if (op->getOpType() == OpType::Add) {
                if (isScalarConstantValue(op->getInput(0), 0.0f)) {
                    replacement = op->getInput(1);
                } else if (isScalarConstantValue(op->getInput(1), 0.0f)) {
                    replacement = op->getInput(0);
                }
            } else if (op->getOpType() == OpType::Sub) {
                if (isScalarConstantValue(op->getInput(1), 0.0f)) {
                    replacement = op->getInput(0);
                }
            } else if (op->getOpType() == OpType::Mul) {
                if (isScalarConstantValue(op->getInput(0), 1.0f)) {
                    replacement = op->getInput(1);
                } else if (isScalarConstantValue(op->getInput(1), 1.0f)) {
                    replacement = op->getInput(0);
                }
            } else if (op->getOpType() == OpType::Div) {
                if (isScalarConstantValue(op->getInput(1), 1.0f)) {
                    replacement = op->getInput(0);
                }
            } else if (op->getOpType() == OpType::Clip) {
                auto minVal = getScalarConstantValue(op->getInput(1));
                auto maxVal = getScalarConstantValue(op->getInput(2));
                if (minVal.has_value() && maxVal.has_value() &&
                    std::isinf(*minVal) && *minVal < 0.0f &&
                    std::isinf(*maxVal) && *maxVal > 0.0f) {
                    replacement = op->getInput(0);
                }
            }

            if (!replacement) {
                continue;
            }

            auto output = op->getOutput(0);
            graph.replaceAllUses(output, replacement);
            graph.eraseOperator(op);
            graph.eraseTensorIfUnused(output);
            changed = true;
        }
        return changed;
    }
};

class IdempotenceEliminationPass final : public GraphPass {
  public:
    string name() const override { return "IdempotenceEliminationPass"; }

    bool run(GraphObj &graph) override {
        bool changed = false;
        auto ops = graph.getOperators();
        for (const auto &op : ops) {
            Tensor replacement = nullptr;
            if (op->getNumInputs() < 1 || op->getNumOutputs() != 1) {
                continue;
            }
            auto input = op->getInput(0);
            if (!input) {
                continue;
            }
            auto source = input->getSource();
            if (!source) {
                continue;
            }

            if (op->getOpType() == OpType::Relu &&
                source->getOpType() == OpType::Relu) {
                replacement = input;
            } else if (op->getOpType() == OpType::Clip &&
                       source->getOpType() == OpType::Clip &&
                       op->getNumInputs() == 3 && source->getNumInputs() == 3 &&
                       op->getInput(1) == source->getInput(1) &&
                       op->getInput(2) == source->getInput(2)) {
                replacement = input;
            }

            if (!replacement) {
                continue;
            }

            auto output = op->getOutput(0);
            graph.replaceAllUses(output, replacement);
            graph.eraseOperator(op);
            graph.eraseTensorIfUnused(output);
            changed = true;
        }
        return changed;
    }
};

class DeadCodeEliminationPass final : public GraphPass {
  public:
    string name() const override { return "DeadCodeEliminationPass"; }

    bool run(GraphObj &graph) override {
        IT_ASSERT(!graph.getOutputs().empty(),
                  "DeadCodeEliminationPass requires explicit graph outputs");

        bool changed = false;
        std::unordered_set<Tensor> liveTensors(graph.getOutputs().begin(),
                                               graph.getOutputs().end());
        std::unordered_set<Operator> liveOps;
        std::queue<Tensor> pending;
        for (const auto &tensor : graph.getOutputs()) {
            pending.push(tensor);
        }

        while (!pending.empty()) {
            auto tensor = pending.front();
            pending.pop();
            if (auto source = tensor->getSource()) {
                if (liveOps.insert(source).second) {
                    for (const auto &input : source->getInputs()) {
                        if (input && liveTensors.insert(input).second) {
                            pending.push(input);
                        }
                    }
                }
            }
        }

        auto ops = graph.getOperators();
        for (const auto &op : ops) {
            if (!liveOps.count(op)) {
                auto opOutputs = op->getOutputs();
                graph.eraseOperator(op);
                for (const auto &tensor : opOutputs) {
                    if (tensor) {
                        graph.eraseTensorIfUnused(tensor);
                    }
                }
                changed = true;
            }
        }

        auto tensors = graph.getTensors();
        for (const auto &tensor : tensors) {
            if (!liveTensors.count(tensor) &&
                graph.eraseTensorIfUnused(tensor)) {
                changed = true;
            }
        }
        return changed;
    }
};

class GraphValidatePass final : public GraphPass {
  public:
    string name() const override { return "GraphValidatePass"; }
    bool run(GraphObj &graph) override { return graph.checkValid(); }
};

} // namespace

void GraphOptimizer::run(GraphObj &graph) const {
    std::vector<std::unique_ptr<GraphPass>> passes;
    passes.emplace_back(std::make_unique<TopologicalSortPass>());
    passes.emplace_back(std::make_unique<ShapeInferPass>());
    passes.emplace_back(std::make_unique<ConstantFoldPass>());
    passes.emplace_back(std::make_unique<IdentityEliminationPass>());
    passes.emplace_back(std::make_unique<IdempotenceEliminationPass>());
    passes.emplace_back(std::make_unique<DeadCodeEliminationPass>());
    passes.emplace_back(std::make_unique<TopologicalSortPass>());
    passes.emplace_back(std::make_unique<ShapeInferPass>());
    passes.emplace_back(std::make_unique<GraphValidatePass>());

    for (const auto &pass : passes) {
        pass->run(graph);
    }
}

} // namespace infini
