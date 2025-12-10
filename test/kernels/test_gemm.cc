#include "core/runtime.h"
#include "operators/Gemm.h"
#include "gtest/gtest.h"

namespace infini
{
    void runGemmTest(const std::string &deviceName,
                     infiniDevice_t DeviceT,
                     const Shape &shapeA, const Shape &shapeB,
                     float alpha, float beta, bool transA, bool transB,
                     const DataType &dataType, bool print = false)
    {
        Runtime &runtime = RuntimeObj::getInstance();
        RuntimeObj::init();
        runtime->initThreadContext(DeviceT, 0);
        Graph g = make_ref<GraphObj>(runtime);
        auto A = g->addTensor(shapeA, dataType);
        auto B = g->addTensor(shapeB, dataType);
        auto op = g->addOp<GemmObj>(A, B, nullptr, nullptr, alpha, beta, transA, transB);
        g->dataMalloc();
        auto res = g->toString();
        // Only when the data is contiguous, will this assignment be successful.
        std::vector<float> inputAData(A->getElement());
        std::iota(inputAData.begin(), inputAData.end(), 1);
        std::vector<float> inputBData(B->getElement());
        std::iota(inputBData.begin(), inputBData.end(), 1);
        A->setData(inputAData.data());
        B->setData(inputBData.data());
        std::cout << res << std::endl;
        runtime->run(g);
        auto output = op->getOutput(0);
        output->printData(runtime);
    }

    TEST(Gemm, Kernel)
    {
        runGemmTest("CPU", INFINI_DEVICE_CPU, Shape{3, 5}, Shape{5, 2}, 1.0, 0.0, false, false, DataType(INFINI_DTYPE_F32));
        runGemmTest("NVIDIA", INFINI_DEVICE_NVIDIA, Shape{3, 5}, Shape{5, 2}, 1.0, 0.0, false, false, DataType(INFINI_DTYPE_F32));
    }
} // namespace infini
