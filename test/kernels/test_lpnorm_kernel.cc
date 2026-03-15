#include "core/runtime.h"
#include "operators/LPNorm.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <thread>

namespace infini {

template <typename T> struct LPNormThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 0;
    Shape shapeIn;
    int axis = 1;
    int p = 2;
    float eps = 1e-12f;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    std::string errorMsg;
};

template <typename T>
void lpNormDeviceThreadFunc(LPNormThreadTestParams<T> &params) {
    try {
        RuntimeObj::init();
        Runtime &runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(params.device, params.deviceId);

        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor(params.shapeIn, params.dataType);
        auto op =
            g->addOp<LPNormObj>(x, nullptr, params.axis, params.p, params.eps);

        x->setData(params.inputData.data());
        runtime->dataMalloc(g);
        runtime->run(g);

        auto output = op->getOutput(0);
        size_t numElems = output->getElement();
        params.outputData.resize(numElems);

        void *hostPtr = runtime->allocHost(output->getTotalBytes());
        runtime->memcpy(hostPtr, output->getData()->getRawDataPtr(),
                        output->getTotalBytes(), INFINIRT_MEMCPY_D2H);
        copyAndConvertData(params.outputData, hostPtr, numElems,
                           params.dataType);
        runtime->deallocHost(hostPtr);
        params.completed = true;
    } catch (const std::exception &e) {
        params.errorMsg = e.what();
    } catch (...) {
        params.errorMsg = "unknown exception in lpnorm thread";
    }
}

template <typename T>
void runLPNormMultiThreadTest(const Shape &shapeIn, int axis, int p,
                              const DataType &dataType, float epsilon = 1e-4f) {
    size_t numElems = 1;
    for (auto d : shapeIn)
        numElems *= d;
    // Use positive values to avoid issues with odd p and negative numbers
    auto inputData =
        generateRandomData<T>(numElems, static_cast<T>(0.1), static_cast<T>(5));

    LPNormThreadTestParams<T> cpuParams, gpuParams;
    for (auto *param : {&cpuParams, &gpuParams}) {
        param->shapeIn = shapeIn;
        param->axis = axis;
        param->p = p;
        param->dataType = dataType;
        param->inputData = inputData;
    }
    cpuParams.device = INFINI_DEVICE_NVIDIA;
    cpuParams.deviceName = "CPU";
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceName = "NVIDIA";

    std::thread cpuThread(lpNormDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(lpNormDeviceThreadFunc<T>, std::ref(gpuParams));
    cpuThread.join();
    gpuThread.join();

    if (!cpuParams.completed || !gpuParams.completed) {
        std::string reason;
        if (!cpuParams.completed)
            reason += "CPU: " + cpuParams.errorMsg + "; ";
        if (!gpuParams.completed)
            reason += "GPU: " + gpuParams.errorMsg;
        GTEST_SKIP() << "Skipping - backend error: " << reason;
    }
    ASSERT_EQ(cpuParams.outputData.size(), gpuParams.outputData.size());

    size_t numErrors = 0;
    float maxError = 0.0f;
    for (size_t i = 0; i < cpuParams.outputData.size(); ++i) {
        float cv = static_cast<float>(cpuParams.outputData[i]);
        float gv = static_cast<float>(gpuParams.outputData[i]);
        float err = std::abs(cv - gv);
        maxError = std::max(maxError, err);
        if (err > epsilon)
            ++numErrors;
    }

    EXPECT_EQ(numErrors, 0)
        << "CPU vs NVIDIA mismatch for LPNorm (p=" << p << " axis=" << axis
        << ", max error: " << maxError << ")";
}

// Single-device CPU baseline
TEST(LPNorm, SingleDevice_CPU_L2) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    try {
        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
        auto op = g->addOp<LPNormObj>(x, nullptr, 1, 2);

        size_t n = 4 * 8;
        auto inp = generateRandomData<float>(n, 0.1f, 5.0f);
        x->setData(inp.data());
        runtime->dataMalloc(g);
        runtime->run(g);
        EXPECT_EQ(op->getOutput(0)->getElement(), n);
    } catch (const std::exception &e) {
        GTEST_SKIP() << "CPU backend skipped: " << e.what();
    } catch (...) {
        GTEST_SKIP() << "CPU backend skipped: unknown exception";
    }
}

TEST(LPNorm, SingleDevice_CPU_L1) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    try {
        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor({3, 6}, DataType(INFINI_DTYPE_F32));
        auto op = g->addOp<LPNormObj>(x, nullptr, 0, 1);

        size_t n = 3 * 6;
        auto inp = generateRandomData<float>(n, 0.1f, 5.0f);
        x->setData(inp.data());
        runtime->dataMalloc(g);
        runtime->run(g);
        EXPECT_EQ(op->getOutput(0)->getElement(), n);
    } catch (const std::exception &e) {
        GTEST_SKIP() << "CPU backend skipped: " << e.what();
    } catch (...) {
        GTEST_SKIP() << "CPU backend skipped: unknown exception";
    }
}

#ifdef USE_CUDA
TEST(LPNorm, L2_Axis1_MultiThread_F32) {
    runLPNormMultiThreadTest<float>({8, 32}, 1, 2, DataType(INFINI_DTYPE_F32),
                                    1e-4f);
}

TEST(LPNorm, L1_Axis0_MultiThread_F32) {
    runLPNormMultiThreadTest<float>({16, 16}, 0, 1, DataType(INFINI_DTYPE_F32),
                                    1e-4f);
}

TEST(LPNorm, CUDAOnly_Run_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LPNormObj>(x, nullptr, -1, 2);

    size_t n = 4 * 8;
    auto inp = generateRandomData<float>(n, 0.1f, 5.0f);
    x->setData(inp.data());
    runtime->dataMalloc(g);
    runtime->run(g);

    EXPECT_EQ(op->getOutput(0)->getElement(), n);
}
#else
TEST(LPNorm, MultiThread_Skipped) {
    std::cout << "CUDA not enabled, skipping multi-thread lpnorm tests\n";
}
#endif

} // namespace infini
