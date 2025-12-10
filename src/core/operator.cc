#include "core/operator.h"
#include "core/graph.h"

namespace infini
{

    OperatorObj::OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs)
        : type(opType), inputs(inputs), outputs(outputs) {}

    const TensorVec &OperatorObj::getInputs() const { return inputs; }

    const TensorVec &OperatorObj::getOutputs() const { return outputs; }

    Tensor OperatorObj::getInput(size_t idx) const
    {
        IT_ASSERT(idx < inputs.size(), "Invalid input index");
        return inputs.at(idx);
    }

    Tensor OperatorObj::getOutput(size_t idx) const
    {
        IT_ASSERT(idx < outputs.size(), "Invalid output index");
        return outputs.at(idx);
    }

    OpVec OperatorObj::getPredecessors() const { return wrefs_to_refs(predecessors); }

    OpVec OperatorObj::getSuccessors() const { return wrefs_to_refs(successors); }

    OpType OperatorObj::getOpType() const { return type; }

    DataType OperatorObj::getInDType(size_t idx) const
    {
        IT_ASSERT(idx < inputs.size(), "Invalid input index");
        return getInput(idx)->getDataType();
    }

    DataType OperatorObj::getOutDType(size_t idx) const
    {
        IT_ASSERT(idx < outputs.size(), "Invalid output index");
        return getOutput(idx)->getDataType();
    }

    ElementType OperatorObj::getNumInputs() const
    {
        return inputs.size();
    }

    ElementType OperatorObj::getNumOutputs() const
    {
        return outputs.size();
    }

    void *OperatorObj::getInfiniOpDesc() const
    {
        return infiniOpDesc;
    };

    void OperatorObj::removePredecessors(const Operator &op)
    {
        for (auto it = predecessors.begin(); it != predecessors.end();)
        {
            if (it->lock() == op)
                it = predecessors.erase(it);
            else
                ++it;
        }
    }

    void OperatorObj::removeSuccessors(const Operator &op)
    {
        for (auto it = successors.begin(); it != successors.end();)
        {
            if (it->lock() == op)
                it = successors.erase(it);
            else
                ++it;
        }
    }

    void OperatorObj::replaceInput(Tensor t1, Tensor t2)
    {
        for (auto itr = inputs.begin(); itr != inputs.end(); ++itr)
        {
            if (*itr == t1)
            {
                *itr = t2;
            }
        }
    }

    void OperatorObj::addPredecessors(const Operator &op) { predecessors.emplace_back(op); }
    void OperatorObj::addSuccessors(const Operator &op) { successors.emplace_back(op); }

    bool OperatorObj::checkValid(GraphObj *graph)
    {
        auto optShapes = inferShape();
        if (!optShapes)
        {
            return false;
        }

        const vector<Shape> &shapes = *optShapes;
        if (shapes.size() != outputs.size())
        {
            return false;
        }
        if (graph)
        { // if graph != nullptr, outputs should be created
            auto dataTypes = inferDataType();
            for (size_t i = 0; i < outputs.size(); ++i)
            {
                IT_ASSERT(!outputs[i], "Find empty output while operator creation");
                outputs[i] = graph->addTensor(shapes[i], dataTypes[i]);
            }
        }
        else
        { // if outputs have been created, check their shapes
            for (size_t i = 0; i < shapes.size(); ++i)
            {
                if (shapes[i] != outputs[i]->getShape())
                {
                    return false;
                }
            }
        }
        return true;
    }

} // namespace infini
