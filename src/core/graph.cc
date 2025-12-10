#include "core/graph.h"

namespace infini
{
    GraphObj::GraphObj(Runtime runtime) : runtime(runtime) {}

    std::string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "=== Graph ===\n";

        oss << "[Tensors]\n";
        for (const auto &tensor : tensors)
        {
            oss << "  " << tensor << "\n";
        }

        oss << "[Operators]\n";
        for (const auto &op : ops)
        {
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
        return oss.str();
    }

    Runtime GraphObj::getRuntime() const { return runtime; }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    void GraphObj::removeOperator(Operator op)
    {
        auto it = std::find(ops.begin(), ops.end(), op);
        if (it != ops.end())
            ops.erase(it);
    }

    void GraphObj::removeTensor(Tensor tensor)
    {
        auto it = std::find(tensors.begin(), tensors.end(), tensor);
        if (it != tensors.end())
            tensors.erase(it);
    }

    const TensorVec &GraphObj::getTensors() const { return tensors; }

    const OpVec &GraphObj::getOperators() const { return ops; }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    bool GraphObj::topo_sort()
    {
        std::unordered_map<OperatorObj *, int> indegree;
        for (auto &op : ops)
        {
            indegree[op.get()] = 0;
        }
        for (auto &op : ops)
        {
            for (auto &input : op->getInputs())
            {
                if (auto src = input->getSource())
                {
                    indegree[op.get()]++;
                }
            }
        }

        std::queue<Operator> q;
        for (auto &op : ops)
        {
            if (indegree[op.get()] == 0)
            {
                q.push(op);
            }
        }

        std::vector<Operator> sorted;
        while (!q.empty())
        {
            auto op = q.front();
            q.pop();
            sorted.push_back(op);

            for (auto &succ : op->getSuccessors())
            {
                if (--indegree[succ.get()] == 0)
                {
                    q.push(succ);
                }
            }
        }

        if (sorted.size() != ops.size())
        {
            // 有环，拓扑失败
            return false;
        }

        ops = std::move(sorted);
        return true;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getShape();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        for (auto &tensor : tensors)
        {
            tensor->dataMalloc(runtime);
        }
    }

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    bool GraphObj::checkValid() const
    {
        // 构建快速查找集合
        std::unordered_set<Tensor> tensorSet(tensors.begin(), tensors.end());
        std::unordered_set<Operator> opSet(ops.begin(), ops.end());

        // 1. 检查所有 Tensor
        for (auto tensor : tensors)
        {
            // 必须有 source 或 targets
            IT_ASSERT(!(tensor->getTargets().empty() && tensor->getSource() == nullptr),
                      "Invalid tensor: " + tensor->toString() +
                          " has no source and no targets");

            // 检查 target ops 是否都在 Graph
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(opSet.count(op),
                          "Tensor " + tensor->toString() +
                              " has target op not in graph: " + op->toString());
            }

            // 检查 source op 是否在 Graph
            if (auto src = tensor->getSource())
            {
                IT_ASSERT(opSet.count(src),
                          "Tensor " + tensor->toString() +
                              " has source op not in graph: " + src->toString());
            }
        }

        // 2. 检查所有 Operator
        for (auto op : ops)
        {
            // 输入 tensor 必须存在
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(tensorSet.count(tensor),
                          "Op " + op->toString() +
                              " has input tensor not in graph: " + tensor->toString());
            }
            // 输出 tensor 必须存在
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(tensorSet.count(tensor),
                          "Op " + op->toString() +
                              " has output tensor not in graph: " + tensor->toString());
            }
            // 前驱/后继必须存在
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(opSet.count(pre),
                          "Op " + op->toString() +
                              " has predecessor not in graph: " + pre->toString());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(opSet.count(suc),
                          "Op " + op->toString() +
                              " has successor not in graph: " + suc->toString());
            }
        }

        // 3. 检查 Tensor 的 FUID 唯一性
        std::unordered_set<UidBaseType> fuids;
        for (auto tensor : tensors)
        {
            IT_ASSERT(fuids.insert(tensor->getFuid()).second,
                      "Duplicate tensor fuid: " + std::to_string(tensor->getFuid()));
        }

        // 4. 检查双向一致性
        for (auto tensor : tensors)
        {
            for (auto targetOp : tensor->getTargets())
            {
                auto &inputs = targetOp->getInputs();
                IT_ASSERT(std::find(inputs.begin(), inputs.end(), tensor) != inputs.end(),
                          "Mismatch: tensor " + tensor->toString() +
                              " lists target op " + targetOp->toString() +
                              " but that op does not have this tensor as input");
            }
            if (auto srcOp = tensor->getSource())
            {
                auto &outputs = srcOp->getOutputs();
                IT_ASSERT(std::find(outputs.begin(), outputs.end(), tensor) != outputs.end(),
                          "Mismatch: tensor " + tensor->toString() +
                              " has source op " + srcOp->toString() +
                              " but that op does not have this tensor as output");
            }
        }

        return true;
    }

} // namespace infini
