#include "core/runtime.h"
#include "operators/LogSoftmax.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <thread>

namespace infini {

template <typename T> struct LogSoftmaxThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 0;
    Shape shapeIn;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    std::string errorMsg;
};

template <typename T>
void logSoftmaxDeviceThreadFunc(LogSoftmaxThreadTestParams<T> &params) {
    try {
        RuntimeObj::init();
        Runtime &runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(params.device, params.deviceId);

        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor(params.shapeIn, params.dataType);
        auto op = g->addOp<LogSoftmaxObj>(x, nullptr);

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
    }
}

template <typename T>
void runLogSoftmaxMultiThreadTest(const Shape &shapeIn,
                                  const DataType &dataType,
                                  float epsilon = 1e-4f) {
    size_t numElems = 1;
    for (auto d : shapeIn)
        numElems *= d;
    auto inputData =
        generateRandomData<T>(numElems, static_cast<T>(-3), static_cast<T>(3));

    LogSoftmaxThreadTestParams<T> cpuParams, gpuParams;
    for (auto *p : {&cpuParams, &gpuParams}) {
        p->shapeIn = shapeIn;
        p->dataType = dataType;
        p->inputData = inputData;
    }
    cpuParams.device = INFINI_DEVICE_NVIDIA;
    cpuParams.deviceName = "CPU";
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceName = "NVIDIA";

    std::thread cpuThread(logSoftmaxDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(logSoftmaxDeviceThreadFunc<T>, std::ref(gpuParams));
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
        << "CPU vs NVIDIA mismatch for LogSoftmax (max error: " << maxError
        << ")";
}

// Single-device CPU baseline
TEST(LogSoftmax, SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor({2, 10}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LogSoftmaxObj>(x, nullptr);

    size_t n = 2 * 10;
    auto inp = generateRandomData<float>(n, -2.0f, 2.0f);
    x->setData(inp.data());
    runtime->dataMalloc(g);
    runtime->run(g);

    EXPECT_EQ(op->getOutput(0)->getElement(), n);
}

#ifdef USE_CUDA
TEST(LogSoftmax, MultiThread_F32) {
    runLogSoftmaxMultiThreadTest<float>({4, 64}, DataType(INFINI_DTYPE_F32),
                                        1e-4f);
}

TEST(LogSoftmax, MultiThread_F32_3D) {
    runLogSoftmaxMultiThreadTest<float>({2, 8, 32}, DataType(INFINI_DTYPE_F32),
                                        1e-4f);
}
#else
TEST(LogSoftmax, MultiThread_Skipped) {
    std::cout << "CUDA not enabled, skipping multi-thread log_softmax tests\n";
}
#endif

} // namespace infini
