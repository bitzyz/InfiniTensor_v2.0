#include "core/runtime.h"
#include "operators/Clip.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

namespace infini {

// Thread test parameters
template <typename T> struct ThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    Shape shapeInput;
    Shape shapeMin;
    Shape shapeMax;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> minData;
    std::vector<T> maxData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
};

// Device thread function
template <typename T> void deviceThreadFunc(ThreadTestParams<T> &params) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();

    // Initialize device Context
    runtime->initThreadContext(params.device, params.deviceId);

    // Create Graph
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(params.shapeInput, params.dataType);
    auto min = g->addTensor(params.shapeMin, params.dataType);
    auto max = g->addTensor(params.shapeMax, params.dataType);
    auto op = g->addOp<ClipObj>(input, nullptr, min, max);

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(params.inputData.data());
    min->setData(params.minData.data());
    max->setData(params.maxData.data());
    // Allocate memory (output only, inputs are externally managed)
    runtime->dataMalloc(g);

    // Track device pointers for cleanup
    void *deviceInput = nullptr;
    void *deviceMin = nullptr;
    void *deviceMax = nullptr;

    // For GPU, manually copy input data from CPU to GPU device memory
    if (params.device != INFINI_DEVICE_CPU) {
        deviceInput = runtime->allocDevice(input->getTotalBytes());
        deviceMin = runtime->allocDevice(min->getTotalBytes());
        deviceMax = runtime->allocDevice(max->getTotalBytes());
        runtime->memcpy(deviceInput, params.inputData.data(),
                        input->getTotalBytes(), INFINIRT_MEMCPY_H2D);
        runtime->memcpy(deviceMin, params.minData.data(), min->getTotalBytes(),
                        INFINIRT_MEMCPY_H2D);
        runtime->memcpy(deviceMax, params.maxData.data(), max->getTotalBytes(),
                        INFINIRT_MEMCPY_H2D);
        input->setData(deviceInput);
        min->setData(deviceMin);
        max->setData(deviceMax);
    }

    // Run computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before copying data
    if (!runtime->isCpu()) {
        runtime->synchronize();
    }

    // Get output and copy to host
    auto output = op->getOutput(0);
    size_t numElements = output->getElement();
    params.outputData.resize(numElements);

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
    if (runtime->isCpu()) {
        // For CPU, data is already in host memory
        copyAndConvertData(params.outputData, devicePtr, numElements,
                           params.dataType);
    } else {
        // For GPU, need to copy from device to host
        void *hostPtr = runtime->allocHost(output->getTotalBytes());
        runtime->memcpy(hostPtr, devicePtr, output->getTotalBytes(),
                        INFINIRT_MEMCPY_D2H);
        if (params.deviceName == "NVIDIA") {
            float *debugPtr = static_cast<float *>(hostPtr);
            std::cout << "DEBUG Clip: First value after memcpy: " << debugPtr[0]
                      << std::endl;
        }
        copyAndConvertData(params.outputData, hostPtr, numElements,
                           params.dataType);
        runtime->deallocHost(hostPtr);
    }

    // Clean up device memory
    if (params.device != INFINI_DEVICE_CPU) {
        runtime->deallocDevice(deviceInput);
        runtime->deallocDevice(deviceMin);
        runtime->deallocDevice(deviceMax);
        // Also clean up output memory allocated by dataMalloc
        runtime->deallocDevice(devicePtr);
        // Clean up workspace to free GPU memory
        auto ctx = runtime->getCurrentThreadContext();
        if (ctx->workspace) {
            runtime->deallocDevice(ctx->workspace);
            ctx->workspace = nullptr;
        }
    }

    params.completed = true;
}

// Data generator function type
template <typename T>
using DataGeneratorFunc = std::function<std::vector<T>(size_t, T, T)>;

// Run multi-thread test
template <typename T>
void runMultiThreadTest(
    const Shape &shapeInput, const Shape &shapeMin, const Shape &shapeMax,
    const DataType &dataType,
    DataGeneratorFunc<T> dataGenerator = generateRandomData<T>,
    bool print = false) {

    // Prepare input data - use utility function to simplify
    size_t elementInput = 1, elementMin = 1, elementMax = 1;
    for (auto dim : shapeInput)
        elementInput *= dim;
    for (auto dim : shapeMin)
        elementMin *= dim;
    for (auto dim : shapeMax)
        elementMax *= dim;

    // Use the passed data generator function (default uses
    // random data)
    auto inputData =
        dataGenerator(elementInput, static_cast<T>(-10), static_cast<T>(10));
    auto minData =
        dataGenerator(elementMin, static_cast<T>(-5), static_cast<T>(0));
    auto maxData =
        dataGenerator(elementMax, static_cast<T>(1), static_cast<T>(5));

    // Create thread parameters
    ThreadTestParams<T> cpuParams, gpuParams;

    // CPU thread parameters
    cpuParams.device = INFINI_DEVICE_CPU;
    cpuParams.deviceId = 0;
    cpuParams.shapeInput = shapeInput;
    cpuParams.shapeMin = shapeMin;
    cpuParams.shapeMax = shapeMax;
    cpuParams.dataType = dataType;
    cpuParams.inputData = inputData;
    cpuParams.minData = minData;
    cpuParams.maxData = maxData;
    cpuParams.deviceName = "CPU";

    // GPU thread parameters
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceId = 5;
    gpuParams.shapeInput = shapeInput;
    gpuParams.shapeMin = shapeMin;
    gpuParams.shapeMax = shapeMax;
    gpuParams.dataType = dataType;
    gpuParams.inputData = inputData;
    gpuParams.minData = minData;
    gpuParams.maxData = maxData;
    gpuParams.deviceName = "NVIDIA";

    if (print) {
        std::cout << "========================================" << std::endl;
        std::cout << "Running Multi-Thread Clip Test" << std::endl;
        std::cout << "DataType: " << dataType.toString() << std::endl;
        std::cout << "Shape Input: " << vecToString(shapeInput) << std::endl;
        std::cout << "Shape Min: " << vecToString(shapeMin) << std::endl;
        std::cout << "Shape Max: " << vecToString(shapeMax) << std::endl;
        std::cout << "Thread 1: CPU (" << dataType.toString() << ")"
                  << std::endl;
        std::cout << "Thread 2: NVIDIA (" << dataType.toString() << ")"
                  << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // Launch two threads for parallel execution
    std::thread cpuThread(deviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(deviceThreadFunc<T>, std::ref(gpuParams));

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
    const float epsilon = 1e-3f;

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
            if (numErrors <= 5) {
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

    EXPECT_EQ(numErrors, 0) << "Results mismatch between "
                               "CPU and NVIDIA (max error: "
                            << maxError << ")";
}

// Basic Clip operation test - F32
TEST(Clip, Clip_MultiThread_F32) {
    Shape shapeInput = {2, 3, 4};
    Shape shapeMin = {2, 3, 4};
    Shape shapeMax = {2, 3, 4};

#ifdef USE_CUDA
    runMultiThreadTest<float>(shapeInput, shapeMin, shapeMax,
                              DataType(INFINI_DTYPE_F32));
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// Basic Clip operation test - F16
TEST(Clip, Clip_MultiThread_F16) {
    Shape shapeInput = {2, 3, 4};
    Shape shapeMin = {2, 3, 4};
    Shape shapeMax = {2, 3, 4};

#ifdef USE_CUDA
    runMultiThreadTest<uint16_t>(shapeInput, shapeMin, shapeMax,
                                 DataType(INFINI_DTYPE_F16),
                                 generateSequentialData<uint16_t>, true);
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// Clip operation test with different shapes - F32
TEST(Clip, Clip_MultiThread_F32_DifferentShapes) {
    Shape shapeInput = {3, 4};
    Shape shapeMin = {3, 4};
    Shape shapeMax = {3, 4};

#ifdef USE_CUDA
    runMultiThreadTest<float>(shapeInput, shapeMin, shapeMax,
                              DataType(INFINI_DTYPE_F32));
#endif
}

// Single device test - CPU
TEST(Clip, Clip_SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    Shape shapeInput = {2, 3, 4};
    Shape shapeMin = {2, 3, 4};
    Shape shapeMax = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto min = g->addTensor(shapeMin, DataType(INFINI_DTYPE_F32));
    auto max = g->addTensor(shapeMax, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ClipObj>(input, nullptr, min, max);

    runtime->dataMalloc(g);

    // Set input data
    std::vector<float> inputData(input->getElement());
    std::vector<float> minData(min->getElement());
    std::vector<float> maxData(max->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 20) - 10); // -10 to 9
    }
    for (size_t i = 0; i < minData.size(); ++i) {
        minData[i] = -2.0f;
    }
    for (size_t i = 0; i < maxData.size(); ++i) {
        maxData[i] = 2.0f;
    }

    input->setData(inputData.data());
    min->setData(minData.data());
    max->setData(maxData.data());

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "CPU Clip Output Data: " << std::endl;
    output->printData(runtime);
}

#ifdef USE_CUDA
// Single device test - NVIDIA F32
TEST(Clip, Clip_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};
    Shape shapeMin = {2, 3, 4};
    Shape shapeMax = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto min = g->addTensor(shapeMin, DataType(INFINI_DTYPE_F32));
    auto max = g->addTensor(shapeMax, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ClipObj>(input, nullptr, min, max);

    // Set input data
    std::vector<float> inputData(input->getElement());
    std::vector<float> minData(min->getElement());
    std::vector<float> maxData(max->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 20) - 10); // -10 to 9
    }
    for (size_t i = 0; i < minData.size(); ++i) {
        minData[i] = -2.0f;
    }
    for (size_t i = 0; i < maxData.size(); ++i) {
        maxData[i] = 2.0f;
    }

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(inputData.data());
    min->setData(minData.data());
    max->setData(maxData.data());
    // Allocate memory (output only, inputs are externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    void *deviceMin = runtime->allocDevice(min->getTotalBytes());
    void *deviceMax = runtime->allocDevice(max->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    runtime->memcpy(deviceMin, minData.data(), min->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    runtime->memcpy(deviceMax, maxData.data(), max->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);
    min->setData(deviceMin);
    max->setData(deviceMax);

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 Clip Output Data: " << std::endl;
    output->printData(runtime);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    runtime->deallocDevice(deviceMin);
    runtime->deallocDevice(deviceMax);
    // Clean up output memory allocated by dataMalloc
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Single device test - NVIDIA F16
TEST(Clip, Clip_SingleDevice_NVIDIA_F16) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};
    Shape shapeMin = {2, 3, 4};
    Shape shapeMax = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F16));
    auto min = g->addTensor(shapeMin, DataType(INFINI_DTYPE_F16));
    auto max = g->addTensor(shapeMax, DataType(INFINI_DTYPE_F16));
    auto op = g->addOp<ClipObj>(input, nullptr, min, max);

    // Set input data
    std::vector<uint16_t> inputData(input->getElement());
    std::vector<uint16_t> minData(min->getElement());
    std::vector<uint16_t> maxData(max->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = fp32_to_fp16(static_cast<float>((i % 20) - 10));
    }
    for (size_t i = 0; i < minData.size(); ++i) {
        minData[i] = fp32_to_fp16(-2.0f);
    }
    for (size_t i = 0; i < maxData.size(); ++i) {
        maxData[i] = fp32_to_fp16(2.0f);
    }

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(inputData.data());
    min->setData(minData.data());
    max->setData(maxData.data());
    // Allocate memory (output only, inputs are externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    void *deviceMin = runtime->allocDevice(min->getTotalBytes());
    void *deviceMax = runtime->allocDevice(max->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    runtime->memcpy(deviceMin, minData.data(), min->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    runtime->memcpy(deviceMax, maxData.data(), max->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);
    min->setData(deviceMin);
    max->setData(deviceMax);

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F16 Clip Output Data: " << std::endl;
    output->printData(runtime);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    runtime->deallocDevice(deviceMin);
    runtime->deallocDevice(deviceMax);
    // Clean up output memory allocated by dataMalloc
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}
#endif

} // namespace infini
