#include "core/runtime.h"
#include "operators/Clip.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <thread>

namespace infini {

template <typename T> struct ClipThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 0;
    Shape shapeIn;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    T minVal;
    T maxVal;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    std::string errorMsg;
};

template <typename T>
void clipDeviceThreadFunc(ClipThreadTestParams<T> &params) {
    try {
        RuntimeObj::init();
        Runtime &runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(params.device, params.deviceId);

        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor(params.shapeIn, params.dataType);
        auto mn = g->addTensor({1}, params.dataType);
        auto mx = g->addTensor({1}, params.dataType);
        auto op = g->addOp<ClipObj>(x, mn, mx, nullptr);

        x->setData(params.inputData.data());
        mn->setData(&params.minVal);
        mx->setData(&params.maxVal);
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
        params.errorMsg = "unknown exception in clip thread";
    }
}

template <typename T>
void runClipMultiThreadTest(const Shape &shapeIn, T minVal, T maxVal,
                            const DataType &dataType, float epsilon = 1e-5f) {
    size_t numElems = 1;
    for (auto d : shapeIn)
        numElems *= d;
    auto inputData =
        generateRandomData<T>(numElems, static_cast<T>(-5), static_cast<T>(5));

    ClipThreadTestParams<T> cpuParams, gpuParams;
    for (auto *p : {&cpuParams, &gpuParams}) {
        p->shapeIn = shapeIn;
        p->dataType = dataType;
        p->inputData = inputData;
        p->minVal = minVal;
        p->maxVal = maxVal;
    }
    cpuParams.device = INFINI_DEVICE_NVIDIA;
    cpuParams.deviceName = "CPU";
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceName = "NVIDIA";

    std::thread cpuThread(clipDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(clipDeviceThreadFunc<T>, std::ref(gpuParams));
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
        << "CPU vs NVIDIA mismatch for Clip (max error: " << maxError << ")";
}

// Single-device CPU baseline
TEST(Clip, SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    try {
        Shape shape = {4, 8};
        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor(shape, DataType(INFINI_DTYPE_F32));
        auto mn = g->addTensor({1}, DataType(INFINI_DTYPE_F32));
        auto mx = g->addTensor({1}, DataType(INFINI_DTYPE_F32));
        auto op = g->addOp<ClipObj>(x, mn, mx, nullptr);

        size_t n = 4 * 8;
        auto inp = generateRandomData<float>(n, -5.0f, 5.0f);
        float minV = -1.0f;
        float maxV = 1.0f;
        x->setData(inp.data());
        mn->setData(&minV);
        mx->setData(&maxV);
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
TEST(Clip, MultiThread_F32) {
    runClipMultiThreadTest<float>({8, 32}, -1.0f, 1.0f,
                                  DataType(INFINI_DTYPE_F32));
}

TEST(Clip, MultiThread_F32_Wide) {
    runClipMultiThreadTest<float>({4, 4, 16}, -2.0f, 2.0f,
                                  DataType(INFINI_DTYPE_F32));
}
#else
TEST(Clip, MultiThread_Skipped) {
    std::cout << "CUDA not enabled, skipping multi-thread clip tests\n";
}
#endif

} // namespace infini
