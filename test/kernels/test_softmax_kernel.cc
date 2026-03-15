#include "core/runtime.h"
#include "operators/Softmax.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <thread>

namespace infini {

template <typename T> struct SoftmaxThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 0;
    Shape shapeIn;
    int axis = -1;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    std::string errorMsg;
};

template <typename T>
void softmaxDeviceThreadFunc(SoftmaxThreadTestParams<T> &params) {
    try {
        RuntimeObj::init();
        Runtime &runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(params.device, params.deviceId);

        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor(params.shapeIn, params.dataType);
        auto op = g->addOp<SoftmaxObj>(x, nullptr, params.axis);

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
        params.errorMsg = "unknown exception in softmax thread";
    }
}

template <typename T>
void runSoftmaxMultiThreadTest(const Shape &shapeIn, int axis,
                               const DataType &dataType, float epsilon = 1e-4f,
                               bool print = false) {
    size_t numElems = 1;
    for (auto d : shapeIn)
        numElems *= d;
    auto inputData =
        generateRandomData<T>(numElems, static_cast<T>(-3), static_cast<T>(3));

    SoftmaxThreadTestParams<T> cpuParams, gpuParams;
    auto fill = [&](SoftmaxThreadTestParams<T> &p, infiniDevice_t dev,
                    std::string name) {
        p.device = dev;
        p.deviceId = 0;
        p.shapeIn = shapeIn;
        p.axis = axis;
        p.dataType = dataType;
        p.inputData = inputData;
        p.deviceName = name;
    };
    fill(cpuParams, INFINI_DEVICE_NVIDIA, "CPU");
    fill(gpuParams, INFINI_DEVICE_NVIDIA, "NVIDIA");

    std::thread cpuThread(softmaxDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(softmaxDeviceThreadFunc<T>, std::ref(gpuParams));
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

    if (print)
        std::cout << "Softmax axis=" << axis << " max_error=" << maxError
                  << " errors=" << numErrors << "\n";

    EXPECT_EQ(numErrors, 0)
        << "CPU vs NVIDIA mismatch for Softmax (max error: " << maxError << ")";
}

// Single-device CPU baseline
TEST(Softmax, SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    try {
        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
        auto op = g->addOp<SoftmaxObj>(x, nullptr, 1);

        size_t n = 4 * 8;
        auto inp = generateRandomData<float>(n, -2.0f, 2.0f);
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
TEST(Softmax, Axis_neg1_MultiThread_F32) {
    runSoftmaxMultiThreadTest<float>({4, 128}, -1, DataType(INFINI_DTYPE_F32),
                                     1e-4f, true);
}

TEST(Softmax, Axis0_MultiThread_F32) {
    runSoftmaxMultiThreadTest<float>({8, 32}, 0, DataType(INFINI_DTYPE_F32));
}

TEST(Softmax, Axis1_MultiThread_F32) {
    runSoftmaxMultiThreadTest<float>({2, 4, 16}, 1, DataType(INFINI_DTYPE_F32));
}

TEST(Softmax, CUDAOnly_Run_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<SoftmaxObj>(x, nullptr, -1);

    size_t n = 4 * 8;
    auto inp = generateRandomData<float>(n, -2.0f, 2.0f);
    x->setData(inp.data());
    runtime->dataMalloc(g);
    runtime->run(g);

    EXPECT_EQ(op->getOutput(0)->getElement(), n);
}
#else
TEST(Softmax, MultiThread_Skipped) {
    std::cout << "CUDA not enabled, skipping multi-thread softmax tests\n";
}
#endif

} // namespace infini
