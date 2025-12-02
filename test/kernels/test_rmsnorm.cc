#include "core/runtime.h"
#include "operators/RMSNorm.h"
#include "gtest/gtest.h"

namespace infini
{
    void runRMSNormTest(const std::string &deviceName,
                     infiniDevice_t DeviceT,
                     const Shape &shapeX,
                     const Shape &shapeW,
                     float epsilon,
                     const DataType &dataType, bool print = false)
    {
        Runtime &runtime = RuntimeObj::getInstance();
        RuntimeObj::init();
        runtime->initThreadContext(DeviceT, 0);
        Graph g = make_ref<GraphObj>(runtime);
        auto X = g->addTensor(shapeX, dataType);
        auto W = g->addTensor(shapeW, dataType);
        auto Y = g->addTensor(shapeX, dataType);
        auto op = g->addOp<RMSNormObj>(X, Y, W);
        g->dataMalloc();
        auto res = g->toString();
        // Only when the data is contiguous, will this assignment be successful.
        std::vector<float> inputXData(X->getElement());
        std::iota(inputXData.begin(), inputXData.end(), 1);
        std::vector<float> inputWData(W->getElement());
        std::iota(inputWData.begin(), inputWData.end(), 1);
        X->setData(inputXData.data());
        W->setData(inputWData.data());
        std::for_each(inputXData.begin(), inputXData.end(), [](float val) {
            std::cout << val << ' ';
        });
        std::cout << res << std::endl;
        runtime->run(g);
        auto output = op->getOutput(0);
        output->printData(runtime);
    }

    TEST(RMSNorm, Kernel)
    {
        runRMSNormTest("CPU", INFINI_DEVICE_CPU, Shape{1, 7}, Shape{7}, 0.0001, DataType(INFINI_DTYPE_F32));
        runRMSNormTest("NVIDIA", INFINI_DEVICE_NVIDIA, Shape{1, 7}, Shape{7}, 0.0001, DataType(INFINI_DTYPE_F32));
    }
} // namespace infini
