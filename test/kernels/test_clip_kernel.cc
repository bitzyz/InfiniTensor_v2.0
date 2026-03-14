#include "core/runtime.h"
#include "operators/Clip.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

namespace infini {

// Thread test parameters
template <typename T> struct ClipThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    Shape inputShape;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    float minVal = 0.0f;
    float maxVal = 1.0f;
    std::vector<T> inputData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
};

// Device thread function
template <typename T>
void clipDeviceThreadFunc(ClipThreadTestParams<T> &params) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();

    // Initialize device Context
    runtime->initThreadContext(params.device, params.deviceId);

    // Create Graph
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(params.inputShape, params.dataType);
    // min_val and max_val must have the same shape as input for InfiniCore Clip
    auto min_val = g->addTensor(params.inputShape, params.dataType);
    auto max_val = g->addTensor(params.inputShape, params.dataType);

    auto op = g->addOp<ClipObj>(input, min_val, max_val, nullptr);

    // Set input data
    input->setData(params.inputData.data());

    // Set min/max values - broadcast to match input shape
    size_t numElements = 1;
    for (auto dim : params.inputShape)
        numElements *= dim;
    std::vector<T> minData(numElements, static_cast<T>(params.minVal));
    std::vector<T> maxData(numElements, static_cast<T>(params.maxVal));
    min_val->setData(minData.data());
    max_val->setData(maxData.data());

    runtime->dataMalloc(g);

    // Run computation
    runtime->run(g);

    // Get output and copy to host
    auto output = op->getOutput(0);
    size_t outputNumElements = output->getElement();
    params.outputData.resize(outputNumElements);

    // Check if output data exists
    auto dataBlob = output->getData();
    if (!dataBlob) {
        throw std::runtime_error("Output data blob is null!");
    }
    void *devicePtr = dataBlob->getRawDataPtr();
    if (!devicePtr && !runtime->isCpu()) {
        throw std::runtime_error(
            "Output device pointer is null on GPU device!");
    }

    // Copy result data
    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    runtime->memcpy(hostPtr, devicePtr, output->getTotalBytes(),
                    INFINIRT_MEMCPY_D2H);

    // Use generic function for data copy and conversion
    copyAndConvertData(params.outputData, hostPtr, outputNumElements,
                       params.dataType);

    runtime->deallocHost(hostPtr);
    params.completed = true;
}

// Data generator function type
template <typename T>
using ClipDataGeneratorFunc = std::function<std::vector<T>(size_t, T, T)>;

// Expected clip result calculation
template <typename T>
std::vector<T> computeExpectedClip(const std::vector<T> &inputData,
                                    float minVal, float maxVal) {
    std::vector<T> expected(inputData.size());
    for (size_t i = 0; i < inputData.size(); ++i) {
        float val = static_cast<float>(inputData[i]);
        float clipped = std::min(std::max(val, minVal), maxVal);
        expected[i] = static_cast<T>(clipped);
    }
    return expected;
}

// Run multi-thread test
template <typename T>
void runClipMultiThreadTest(
    const Shape &inputShape, float minVal, float maxVal,
    const DataType &dataType,
    ClipDataGeneratorFunc<T> dataGenerator = generateRandomData<T>,
    bool print = false) {

    // Prepare input data
    size_t numElements = 1;
    for (auto dim : inputShape)
        numElements *= dim;

    // Use the passed data generator function
    auto inputData = dataGenerator(numElements, static_cast<T>(-10),
                                   static_cast<T>(10));

    // Create thread parameters
    ClipThreadTestParams<T> cpuParams, gpuParams;

    // CPU thread parameters
    cpuParams.device = INFINI_DEVICE_CPU;
    cpuParams.deviceId = 0;
    cpuParams.inputShape = inputShape;
    cpuParams.dataType = dataType;
    cpuParams.minVal = minVal;
    cpuParams.maxVal = maxVal;
    cpuParams.inputData = inputData;
    cpuParams.deviceName = "CPU";

    // GPU thread parameters
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceId = 0;
    gpuParams.inputShape = inputShape;
    gpuParams.dataType = dataType;
    gpuParams.minVal = minVal;
    gpuParams.maxVal = maxVal;
    gpuParams.inputData = inputData;
    gpuParams.deviceName = "NVIDIA";

    if (print) {
        std::cout << "========================================" << std::endl;
        std::cout << "Running Multi-Thread Clip Test" << std::endl;
        std::cout << "DataType: " << dataType.toString() << std::endl;
        std::cout << "Input Shape: " << vecToString(inputShape) << std::endl;
        std::cout << "Min: " << minVal << ", Max: " << maxVal << std::endl;
        std::cout << "Thread 1: CPU (" << dataType.toString() << ")"
                  << std::endl;
        std::cout << "Thread 2: NVIDIA (" << dataType.toString() << ")"
                  << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // Launch two threads for parallel execution
    std::thread cpuThread(clipDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(clipDeviceThreadFunc<T>, std::ref(gpuParams));

    // Wait for both threads to complete
    cpuThread.join();
    gpuThread.join();

    // Verify results
    ASSERT_TRUE(cpuParams.completed) << "CPU thread failed";
    ASSERT_TRUE(gpuParams.completed) << "NVIDIA thread failed";

    ASSERT_EQ(cpuParams.outputData.size(), gpuParams.outputData.size())
        << "Output size mismatch";

    // Compare results
    size_t numErrors = 0;
    float maxError = 0.0f;
    const float epsilon = 1e-2f;

    for (size_t i = 0; i < cpuParams.outputData.size(); ++i) {
        float cpuVal, gpuVal;

        // Convert to float for comparison
        if constexpr (std::is_same_v<T, float>) {
            cpuVal = cpuParams.outputData[i];
            gpuVal = gpuParams.outputData[i];
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            // FP16 to FP32 comparison
            cpuVal = fp16_to_fp32(cpuParams.outputData[i]);
            gpuVal = fp16_to_fp32(gpuParams.outputData[i]);
        }

        float error = std::abs(cpuVal - gpuVal);
        maxError = std::max(maxError, error);

        if (error > epsilon) {
            numErrors++;
            if (numErrors <= 5) { // Only print first 5 errors
                std::cout << "Mismatch at index " << i << ": CPU=" << cpuVal
                          << ", NVIDIA=" << gpuVal << ", error=" << error
                          << std::endl;
            }
        }
    }

    if (print) {
        std::cout << "Result Comparison:" << std::endl;
        std::cout << "  Total elements: " << cpuParams.outputData.size()
                  << std::endl;
        std::cout << "  Errors: " << numErrors << std::endl;
        std::cout << "  Max error: " << maxError << std::endl;

        if (numErrors == 0) {
            std::cout << "  Test PASSED" << std::endl;
        } else {
            std::cout << "  Test FAILED" << std::endl;
        }
        std::cout << "========================================" << std::endl;
    }

    EXPECT_EQ(numErrors, 0)
        << "Results mismatch between CPU and NVIDIA (max error: " << maxError
        << ")";
}

// Basic Clip operation test - F32
TEST(Clip, Basic_MultiThread_F32) {
    Shape inputShape = {3, 4};
    float minVal = 2.0f;
    float maxVal = 7.0f;

#ifdef USE_CUDA
    runClipMultiThreadTest<float>(inputShape, minVal, maxVal,
                                  DataType(INFINI_DTYPE_F32),
                                  generateSequentialData<float>, true);
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// Basic Clip operation test - F16
TEST(Clip, Basic_MultiThread_F16) {
    Shape inputShape = {3, 4};
    float minVal = 2.0f;
    float maxVal = 7.0f;

#ifdef USE_CUDA
    runClipMultiThreadTest<uint16_t>(inputShape, minVal, maxVal,
                                     DataType(INFINI_DTYPE_F16),
                                     generateSequentialData<uint16_t>, true);
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// Clip with negative min value - F32
TEST(Clip, NegativeMin_MultiThread_F32) {
    Shape inputShape = {4, 5};
    float minVal = -5.0f;
    float maxVal = 5.0f;

#ifdef USE_CUDA
    runClipMultiThreadTest<float>(inputShape, minVal, maxVal,
                                  DataType(INFINI_DTYPE_F32),
                                  generateRandomData<float>);
#endif
}

// Clip with large values - F32
TEST(Clip, LargeValues_MultiThread_F32) {
    Shape inputShape = {2, 8};
    float minVal = -100.0f;
    float maxVal = 100.0f;

#ifdef USE_CUDA
    runClipMultiThreadTest<float>(inputShape, minVal, maxVal,
                                  DataType(INFINI_DTYPE_F32),
                                  generateRandomData<float>);
#endif
}

// Single device test - CPU
TEST(Clip, SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    Shape inputShape = {3, 4};
    float minVal = 2.0f;
    float maxVal = 7.0f;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(inputShape, DataType(INFINI_DTYPE_F32));
    // min_val and max_val must have the same shape as input for InfiniCore Clip
    auto min_val = g->addTensor(inputShape, DataType(INFINI_DTYPE_F32));
    auto max_val = g->addTensor(inputShape, DataType(INFINI_DTYPE_F32));

    auto op = g->addOp<ClipObj>(input, min_val, max_val, nullptr);

    // Set input data
    size_t numElements = input->getElement();
    std::vector<float> inputData(numElements);
    for (size_t i = 0; i < numElements; ++i) {
        // Some values below min, some above max, some in between
        inputData[i] = static_cast<float>(i) - 5.0f;
    }

    // Set min/max values - broadcast to match input shape
    std::vector<float> minData(numElements, minVal);
    std::vector<float> maxData(numElements, maxVal);
    input->setData(inputData.data());
    min_val->setData(minData.data());
    max_val->setData(maxData.data());

    runtime->dataMalloc(g);

    // Execute computation
    runtime->run(g);

    // Get output and verify
    auto output = op->getOutput(0);
    std::cout << "Input Data: " << std::endl;
    input->printData(runtime);
    std::cout << "Clip(" << minVal << ", " << maxVal << ") Output Data: "
              << std::endl;
    output->printData(runtime);

    // Verify expected values
    auto expected = computeExpectedClip(inputData, minVal, maxVal);
    std::vector<float> outputData(numElements);

    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    auto dataBlob = output->getData();
    runtime->memcpy(hostPtr, dataBlob->getRawDataPtr(), output->getTotalBytes(),
                    INFINIRT_MEMCPY_D2H);
    copyAndConvertData(outputData, hostPtr, numElements,
                       DataType(INFINI_DTYPE_F32));
    runtime->deallocHost(hostPtr);

    // Check results
    size_t errors = 0;
    for (size_t i = 0; i < numElements; ++i) {
        if (std::abs(outputData[i] - expected[i]) > 1e-5f) {
            errors++;
            if (errors <= 5) {
                std::cout << "Error at index " << i << ": expected="
                          << expected[i] << ", got=" << outputData[i]
                          << std::endl;
            }
        }
    }

    EXPECT_EQ(errors, 0) << "CPU clip computation failed with " << errors
                         << " errors";
}

#ifdef USE_CUDA
// Single device test - NVIDIA F32
TEST(Clip, SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape inputShape = {3, 4};
    float minVal = 2.0f;
    float maxVal = 7.0f;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(inputShape, DataType(INFINI_DTYPE_F32));
    // min_val and max_val must have the same shape as input for InfiniCore Clip
    auto min_val = g->addTensor(inputShape, DataType(INFINI_DTYPE_F32));
    auto max_val = g->addTensor(inputShape, DataType(INFINI_DTYPE_F32));

    auto op = g->addOp<ClipObj>(input, min_val, max_val, nullptr);

    // Set input data
    size_t numElements = input->getElement();
    std::vector<float> inputData(numElements);
    for (size_t i = 0; i < numElements; ++i) {
        inputData[i] = static_cast<float>(i) - 5.0f;
    }

    // Set min/max values - broadcast to match input shape
    std::vector<float> minData(numElements, minVal);
    std::vector<float> maxData(numElements, maxVal);
    input->setData(inputData.data());
    min_val->setData(minData.data());
    max_val->setData(maxData.data());

    runtime->dataMalloc(g);

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 Output Data: " << std::endl;
    output->printData(runtime);
}

// Single device test - NVIDIA F16
TEST(Clip, SingleDevice_NVIDIA_F16) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape inputShape = {3, 4};
    float minVal = 2.0f;
    float maxVal = 7.0f;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(inputShape, DataType(INFINI_DTYPE_F16));
    // min_val and max_val must have the same shape as input for InfiniCore Clip
    auto min_val = g->addTensor(inputShape, DataType(INFINI_DTYPE_F16));
    auto max_val = g->addTensor(inputShape, DataType(INFINI_DTYPE_F16));

    auto op = g->addOp<ClipObj>(input, min_val, max_val, nullptr);

    // Set input data
    size_t numElements = input->getElement();
    std::vector<uint16_t> inputData(numElements);
    for (size_t i = 0; i < numElements; ++i) {
        // Generate sequential values that will test clipping
        inputData[i] = fp32_to_fp16(static_cast<float>(i) - 5.0f);
    }

    // Set min/max values - broadcast to match input shape
    std::vector<uint16_t> minData(numElements, fp32_to_fp16(minVal));
    std::vector<uint16_t> maxData(numElements, fp32_to_fp16(maxVal));
    input->setData(inputData.data());
    min_val->setData(minData.data());
    max_val->setData(maxData.data());

    runtime->dataMalloc(g);

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F16 Output Data: " << std::endl;
    output->printData(runtime);
}
#endif

} // namespace infini