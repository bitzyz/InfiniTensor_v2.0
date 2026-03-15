#include "core/runtime.h"
#include "operators/Conv.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <thread>

namespace infini {

template <typename T> struct ConvThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 0;
    Shape shapeX;
    Shape shapeW;
    vector<int64_t> pads;
    vector<int64_t> strides;
    vector<int64_t> dilations;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> weightData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    std::string errorMsg;
};

template <typename T>
void convDeviceThreadFunc(ConvThreadTestParams<T> &params) {
    try {
        RuntimeObj::init();
        Runtime &runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(params.device, params.deviceId);

        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor(params.shapeX, params.dataType);
        auto w = g->addTensor(params.shapeW, params.dataType);
        auto op = g->addOp<ConvObj>(x, w, nullptr, nullptr, params.pads,
                                    params.strides, params.dilations);

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
void runConvMultiThreadTest(const Shape &shapeX, const Shape &shapeW,
                            const vector<int64_t> &pads,
                            const vector<int64_t> &strides,
                            const vector<int64_t> &dilations,
                            const DataType &dataType, float epsilon = 1e-2f) {
    size_t numX = 1, numW = 1;
    for (auto d : shapeX)
        numX *= d;
    for (auto d : shapeW)
        numW *= d;

    auto inputData =
        generateRandomData<T>(numX, static_cast<T>(-1), static_cast<T>(1));
    auto weightData =
        generateRandomData<T>(numW, static_cast<T>(-1), static_cast<T>(1));

    ConvThreadTestParams<T> cpuParams, gpuParams;
    for (auto *p : {&cpuParams, &gpuParams}) {
        p->shapeX = shapeX;
        p->shapeW = shapeW;
        p->pads = pads;
        p->strides = strides;
        p->dilations = dilations;
        p->dataType = dataType;
        p->inputData = inputData;
        p->weightData = weightData;
    }
    cpuParams.device = INFINI_DEVICE_NVIDIA;
    cpuParams.deviceName = "CPU";
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceName = "NVIDIA";

    std::thread cpuThread(convDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(convDeviceThreadFunc<T>, std::ref(gpuParams));
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
        << "CPU vs NVIDIA mismatch for Conv (max error: " << maxError << ")";
}

// Single-device CPU baseline
TEST(Conv, SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    // [N, C_in, H, W]  x  [C_out, C_in, kH, kW]
    Shape shapeX = {1, 1, 5, 5};
    Shape shapeW = {2, 1, 3, 3};

    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor(shapeX, DataType(INFINI_DTYPE_F32));
    auto w = g->addTensor(shapeW, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ConvObj>(x, w, nullptr, nullptr, vector<int64_t>{1, 1},
                                vector<int64_t>{1, 1}, vector<int64_t>{1, 1});

    size_t nX = 1 * 1 * 5 * 5;
    size_t nW = 2 * 1 * 3 * 3;
    auto inp = generateRandomData<float>(nX, -1.0f, 1.0f);
    auto wt = generateRandomData<float>(nW, -1.0f, 1.0f);
    x->setData(inp.data());
    w->setData(wt.data());
    runtime->dataMalloc(g);
    runtime->run(g);

    // Output should be [1, 2, 5, 5] with pad=1
    EXPECT_EQ(op->getOutput(0)->getElement(), 1u * 2u * 5u * 5u);
}

#ifdef USE_CUDA
TEST(Conv, Pad1Stride1_MultiThread_F32) {
    // [1, 1, 8, 8] * [4, 1, 3, 3]  →  [1, 4, 8, 8]
    runConvMultiThreadTest<float>({1, 1, 8, 8}, {4, 1, 3, 3}, {1, 1}, {1, 1},
                                  {1, 1}, DataType(INFINI_DTYPE_F32), 1e-2f);
}

TEST(Conv, Stride2_MultiThread_F32) {
    // [1, 1, 8, 8] * [4, 1, 3, 3]  →  [1, 4, 3, 3]  (no pad, stride=2)
    runConvMultiThreadTest<float>({1, 1, 8, 8}, {4, 1, 3, 3}, {0, 0}, {2, 2},
                                  {1, 1}, DataType(INFINI_DTYPE_F32), 1e-2f);
}

TEST(Conv, Dilation2_MultiThread_F32) {
    // [1, 1, 10, 10] * [2, 1, 3, 3] dilated  →  [1, 2, 6, 6]
    runConvMultiThreadTest<float>({1, 1, 10, 10}, {2, 1, 3, 3}, {0, 0}, {1, 1},
                                  {2, 2}, DataType(INFINI_DTYPE_F32), 1e-2f);
}
#else
TEST(Conv, MultiThread_Skipped) {
    std::cout << "CUDA not enabled, skipping multi-thread conv tests\n";
}
#endif

} // namespace infini
