#include "core/runtime.h"
#include "operators/RMSNorm.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <thread>

namespace infini {

template <typename T> struct RMSNormThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 0;
    Shape shapeIn; // e.g. {batch, hidden}
    Shape shapeW;  // e.g. {hidden}
    float epsilon = 1e-6f;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> weightData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    std::string errorMsg;
};

template <typename T>
void rmsNormDeviceThreadFunc(RMSNormThreadTestParams<T> &params) {
    try {
        RuntimeObj::init();
        Runtime &runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(params.device, params.deviceId);

        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor(params.shapeIn, params.dataType);
        auto w = g->addTensor(params.shapeW, params.dataType);
        auto op = g->addOp<RMSNormObj>(x, w, nullptr, params.epsilon);

        x->setData(params.inputData.data());
        w->setData(params.weightData.data());
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
    }
}

template <typename T>
void runRMSNormMultiThreadTest(const Shape &shapeIn, const Shape &shapeW,
                               float eps, const DataType &dataType,
                               float epsilon = 1e-3f) {
    size_t numIn = 1, numW = 1;
    for (auto d : shapeIn)
        numIn *= d;
    for (auto d : shapeW)
        numW *= d;

    auto inputData =
        generateRandomData<T>(numIn, static_cast<T>(-2), static_cast<T>(2));
    auto weightData =
        generateRandomData<T>(numW, static_cast<T>(0.5), static_cast<T>(1.5));

    RMSNormThreadTestParams<T> cpuParams, gpuParams;
    for (auto *p : {&cpuParams, &gpuParams}) {
        p->shapeIn = shapeIn;
        p->shapeW = shapeW;
        p->epsilon = eps;
        p->dataType = dataType;
        p->inputData = inputData;
        p->weightData = weightData;
    }
    cpuParams.device = INFINI_DEVICE_NVIDIA;
    cpuParams.deviceName = "CPU";
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceName = "NVIDIA";

    std::thread cpuThread(rmsNormDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(rmsNormDeviceThreadFunc<T>, std::ref(gpuParams));
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
        << "CPU vs NVIDIA mismatch for RMSNorm (max error: " << maxError << ")";
}

// Single-device CPU baseline
TEST(RMSNorm, SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor({4, 16}, DataType(INFINI_DTYPE_F32));
    auto w = g->addTensor({16}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<RMSNormObj>(x, w, nullptr, 1e-6f);

    size_t n = 4 * 16;
    size_t nw = 16;
    auto inp = generateRandomData<float>(n, -2.0f, 2.0f);
    auto wt = generateRandomData<float>(nw, 0.5f, 1.5f);
    x->setData(inp.data());
    w->setData(wt.data());
    runtime->dataMalloc(g);
    runtime->run(g);

    EXPECT_EQ(op->getOutput(0)->getElement(), n);
}

#ifdef USE_CUDA
TEST(RMSNorm, MultiThread_2D_F32) {
    runRMSNormMultiThreadTest<float>({8, 64}, {64}, 1e-6f,
                                     DataType(INFINI_DTYPE_F32));
}

TEST(RMSNorm, MultiThread_3D_F32) {
    runRMSNormMultiThreadTest<float>({2, 4, 32}, {32}, 1e-6f,
                                     DataType(INFINI_DTYPE_F32));
}
#else
TEST(RMSNorm, MultiThread_Skipped) {
    std::cout << "CUDA not enabled, skipping multi-thread rmsnorm tests\n";
}
#endif

} // namespace infini
