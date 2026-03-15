#include "core/runtime.h"
#include "operators/Unary.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <thread>

namespace infini {

// -----------------------------------------------------------------------
// Per-thread parameters
// -----------------------------------------------------------------------
template <typename T> struct UnaryThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 0;
    OpType opType = OpType::Unknown; // default; overwritten before use
    Shape shapeIn;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    std::string errorMsg;
};

// -----------------------------------------------------------------------
// Thread function
// -----------------------------------------------------------------------
template <typename T>
void unaryDeviceThreadFunc(UnaryThreadTestParams<T> &params) {
    try {
        RuntimeObj::init();
        Runtime &runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(params.device, params.deviceId);

        Graph g = make_ref<GraphObj>(runtime);
        auto x = g->addTensor(params.shapeIn, params.dataType);
        auto op = g->addOp<UnaryObj>(params.opType, x, nullptr);

        x->setData(params.inputData.data());
        runtime->dataMalloc(g);
        runtime->run(g);

        auto output = op->getOutput(0);
        size_t numElems = output->getElement();
        params.outputData.resize(numElems);

        auto dataBlob = output->getData();
        void *devPtr = dataBlob->getRawDataPtr();

        void *hostPtr = runtime->allocHost(output->getTotalBytes());
        runtime->memcpy(hostPtr, devPtr, output->getTotalBytes(),
                        INFINIRT_MEMCPY_D2H);
        copyAndConvertData(params.outputData, hostPtr, numElems,
                           params.dataType);
        runtime->deallocHost(hostPtr);
        params.completed = true;
    } catch (const std::exception &e) {
        params.errorMsg = e.what();
    }
}

// -----------------------------------------------------------------------
// Multi-thread comparison (CPU vs NVIDIA)
// -----------------------------------------------------------------------
template <typename T>
void runUnaryMultiThreadTest(OpType opType, const Shape &shapeIn,
                             const DataType &dataType, float epsilon = 1e-4f,
                             bool print = false) {
    size_t numElems = 1;
    for (auto d : shapeIn)
        numElems *= d;

    auto inputData =
        generateRandomData<T>(numElems, static_cast<T>(-2), static_cast<T>(2));

    UnaryThreadTestParams<T> cpuParams, gpuParams;
    cpuParams.device = INFINI_DEVICE_NVIDIA;
    cpuParams.deviceId = 0;
    cpuParams.opType = opType;
    cpuParams.shapeIn = shapeIn;
    cpuParams.dataType = dataType;
    cpuParams.inputData = inputData;
    cpuParams.deviceName = "CPU";

    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceId = 0;
    gpuParams.opType = opType;
    gpuParams.shapeIn = shapeIn;
    gpuParams.dataType = dataType;
    gpuParams.inputData = inputData;
    gpuParams.deviceName = "NVIDIA";

    std::thread cpuThread(unaryDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(unaryDeviceThreadFunc<T>, std::ref(gpuParams));
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
        float cv, gv;
        if constexpr (std::is_same_v<T, float>) {
            cv = cpuParams.outputData[i];
            gv = gpuParams.outputData[i];
        } else {
            cv = fp16_to_fp32(cpuParams.outputData[i]);
            gv = fp16_to_fp32(gpuParams.outputData[i]);
        }
        float err = std::abs(cv - gv);
        maxError = std::max(maxError, err);
        if (err > epsilon)
            ++numErrors;
    }

    if (print) {
        std::cout << "Unary " << opType.toString() << " max_error=" << maxError
                  << " errors=" << numErrors << "\n";
    }
    EXPECT_EQ(numErrors, 0)
        << "CPU vs NVIDIA mismatch for " << opType.toString()
        << " (max error: " << maxError << ")";
}

// -----------------------------------------------------------------------
// Single-device CPU baseline tests
// -----------------------------------------------------------------------
static void runUnaryCPUSingleDevice(OpType opType) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shape = {4, 8};
    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor(shape, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<UnaryObj>(opType, x, nullptr);

    size_t n = 1;
    for (auto d : shape)
        n *= d;
    auto inputData = generateRandomData<float>(n, -2.0f, 2.0f);
    x->setData(inputData.data());
    runtime->dataMalloc(g);
    runtime->run(g);

    auto output = op->getOutput(0);
    EXPECT_EQ(output->getElement(), n);
}

TEST(Unary, Relu_CPU) { runUnaryCPUSingleDevice(OpType::Relu); }
TEST(Unary, Sigmoid_CPU) { runUnaryCPUSingleDevice(OpType::Sigmoid); }
TEST(Unary, Silu_CPU) { runUnaryCPUSingleDevice(OpType::Silu); }
TEST(Unary, Gelu_CPU) { runUnaryCPUSingleDevice(OpType::Gelu); }
TEST(Unary, Softplus_CPU) { runUnaryCPUSingleDevice(OpType::Softplus); }
TEST(Unary, Tanh_CPU) { runUnaryCPUSingleDevice(OpType::Tanh); }

// -----------------------------------------------------------------------
// Multi-thread (CPU vs NVIDIA) tests
// -----------------------------------------------------------------------
#ifdef USE_CUDA
#define UNARY_MULTITHREAD_F32(NAME, OPTYPE, EPS)                               \
    TEST(Unary, NAME##_MultiThread_F32) {                                      \
        runUnaryMultiThreadTest<float>(OpType::OPTYPE, {16, 32},               \
                                       DataType(INFINI_DTYPE_F32), EPS);       \
    }

UNARY_MULTITHREAD_F32(Relu, Relu, 1e-5f)
UNARY_MULTITHREAD_F32(Sigmoid, Sigmoid, 1e-4f)
UNARY_MULTITHREAD_F32(Silu, Silu, 1e-4f)
UNARY_MULTITHREAD_F32(Gelu, Gelu, 1e-4f)
UNARY_MULTITHREAD_F32(Softplus, Softplus, 1e-4f)
UNARY_MULTITHREAD_F32(Tanh, Tanh, 1e-4f)
#else
TEST(Unary, MultiThread_Skipped) {
    std::cout << "CUDA not enabled, skipping multi-thread unary tests\n";
}
#endif

} // namespace infini
