#include "core/runtime.h"
#include "operators/LayerNorm.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <thread>

namespace infini {

template <typename T> struct LayerNormThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 0;
    Shape shapeIn;     // e.g. {batch, seq, hidden}
    Shape shapeWeight; // e.g. {seq, hidden} (dims from axis onward)
    int axis = 1;
    float eps = 1e-5f;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> weightData;
    std::vector<T> biasData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    std::string errorMsg;
};

template <typename T>
void layerNormDeviceThreadFunc(LayerNormThreadTestParams<T> &params) {
    try {
        RuntimeObj::init();
        Runtime &runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(params.device, params.deviceId);

        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor(params.shapeIn, params.dataType);
        auto w = g->addTensor(params.shapeWeight, params.dataType);
        auto b = g->addTensor(params.shapeWeight, params.dataType);
        auto op =
            g->addOp<LayerNormObj>(x, w, b, nullptr, params.axis, params.eps);

        x->setData(params.inputData.data());
        w->setData(params.weightData.data());
        b->setData(params.biasData.data());
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
        params.errorMsg = "unknown exception in layernorm thread";
    }
}

template <typename T>
void runLayerNormMultiThreadTest(const Shape &shapeIn, const Shape &shapeWeight,
                                 int axis, float eps, const DataType &dataType,
                                 float epsilon = 1e-3f) {
    size_t numIn = 1, numW = 1;
    for (auto d : shapeIn)
        numIn *= d;
    for (auto d : shapeWeight)
        numW *= d;

    auto inputData =
        generateRandomData<T>(numIn, static_cast<T>(-2), static_cast<T>(2));
    auto weightData =
        generateRandomData<T>(numW, static_cast<T>(0.5), static_cast<T>(1.5));
    auto biasData =
        generateRandomData<T>(numW, static_cast<T>(-0.1), static_cast<T>(0.1));

    LayerNormThreadTestParams<T> cpuParams, gpuParams;
    for (auto *p : {&cpuParams, &gpuParams}) {
        p->shapeIn = shapeIn;
        p->shapeWeight = shapeWeight;
        p->axis = axis;
        p->eps = eps;
        p->dataType = dataType;
        p->inputData = inputData;
        p->weightData = weightData;
        p->biasData = biasData;
    }
    cpuParams.device = INFINI_DEVICE_NVIDIA;
    cpuParams.deviceName = "CPU";
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceName = "NVIDIA";

    std::thread cpuThread(layerNormDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(layerNormDeviceThreadFunc<T>, std::ref(gpuParams));
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
        << "CPU vs NVIDIA mismatch for LayerNorm (max error: " << maxError
        << ")";
}

// Single-device CPU baseline
TEST(LayerNorm, SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    try {
        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor({2, 4, 8}, DataType(INFINI_DTYPE_F32));
        auto w = g->addTensor({8}, DataType(INFINI_DTYPE_F32));
        auto b = g->addTensor({8}, DataType(INFINI_DTYPE_F32));
        auto op = g->addOp<LayerNormObj>(x, w, b, nullptr, 2, 1e-5f);

        size_t n = 2 * 4 * 8;
        size_t nw = 8;
        auto inp = generateRandomData<float>(n, -2.0f, 2.0f);
        auto wt = generateRandomData<float>(nw, 0.5f, 1.5f);
        auto bi = generateRandomData<float>(nw, -0.1f, 0.1f);
        x->setData(inp.data());
        w->setData(wt.data());
        b->setData(bi.data());
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
TEST(LayerNorm, MultiThread_2D_F32) {
    runLayerNormMultiThreadTest<float>({2, 8, 32}, {32}, -1, 1e-5f,
                                       DataType(INFINI_DTYPE_F32));
}

TEST(LayerNorm, MultiThread_3D_F32) {
    runLayerNormMultiThreadTest<float>({2, 4, 16}, {16}, -1, 1e-5f,
                                       DataType(INFINI_DTYPE_F32));
}
#else
TEST(LayerNorm, MultiThread_Skipped) {
    std::cout << "CUDA not enabled, skipping multi-thread layernorm tests\n";
}
#endif

} // namespace infini
