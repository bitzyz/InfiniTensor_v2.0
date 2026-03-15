#include "core/runtime.h"
#include "operators/Conv.h"
#include "gtest/gtest.h"
#include <cstring>

namespace infini {

// Single device test - CPU
TEST(Conv, Conv_SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    // Input shape: [N, C_in, H, W] = [1, 3, 5, 5]
    // Weight shape: [C_out, C_in, Kh, Kw] = [2, 3, 3, 3]
    // Output shape: [N, C_out, H_out, W_out]
    // H_out = (H + 2*pad - dilation*(Kh-1) - 1) / stride + 1
    //       = (5 + 2*1 - 1*(3-1) - 1) / 1 + 1 = 5
    Shape shapeInput = {1, 3, 5, 5};
    Shape shapeWeight = {2, 3, 3, 3};
    Shape shapeBias = {2};
    vector<size_t> strides = {1, 1};
    vector<size_t> dilations = {1, 1};
    vector<size_t> paddings = {1, 1};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto weight = g->addTensor(shapeWeight, DataType(INFINI_DTYPE_F32));
    auto bias = g->addTensor(shapeBias, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ConvObj>(input, weight, bias, strides, dilations,
                                paddings, nullptr);

    runtime->dataMalloc(g);

    // Set input data
    std::vector<float> inputData(input->getElement());
    std::vector<float> weightData(weight->getElement());
    std::vector<float> biasData(bias->getElement());

    // Initialize with simple values
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = 1.0f;
    }
    for (size_t i = 0; i < weightData.size(); ++i) {
        weightData[i] = 0.1f;
    }
    for (size_t i = 0; i < biasData.size(); ++i) {
        biasData[i] = 0.5f;
    }

    input->setData(inputData.data());
    weight->setData(weightData.data());
    bias->setData(biasData.data());

    // Execute computation
    runtime->run(g);

    // Get output and verify
    auto output = op->getOutput(0);
    auto outShape = output->getShape();

    // Verify output shape: [1, 2, 5, 5]
    ASSERT_EQ(outShape->size(), 4);
    EXPECT_EQ((*outShape)[0]->asConstant(), 1);
    EXPECT_EQ((*outShape)[1]->asConstant(), 2);
    EXPECT_EQ((*outShape)[2]->asConstant(), 5);
    EXPECT_EQ((*outShape)[3]->asConstant(), 5);

    std::vector<float> outputData(output->getElement());
    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    auto dataBlob = output->getData();
    runtime->memcpy(hostPtr, dataBlob->getRawDataPtr(), output->getTotalBytes(),
                    INFINIRT_MEMCPY_D2H);
    std::memcpy(outputData.data(), hostPtr, output->getTotalBytes());
    runtime->deallocHost(hostPtr);

    for (size_t i = 0; i < outputData.size(); ++i) {
        EXPECT_GT(outputData[i], 0.0f)
            << "Output at index " << i << " is zero or negative";
    }

    std::cout << "CPU Conv Test PASSED" << std::endl;
}

TEST(Conv, Conv_NoBias_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    Shape shapeInput = {1, 1, 4, 4};
    Shape shapeWeight = {1, 1, 2, 2};
    vector<size_t> strides = {1, 1};
    vector<size_t> dilations = {1, 1};
    vector<size_t> paddings = {0, 0};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto weight = g->addTensor(shapeWeight, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ConvObj>(input, weight, nullptr, strides, dilations,
                                paddings, nullptr);

    runtime->dataMalloc(g);

    // Set input data - simple pattern
    std::vector<float> inputData = {1, 2,  3,  4,  5,  6,  7,  8,
                                    9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> weightData = {1, 1, 1, 1};

    input->setData(inputData.data());
    weight->setData(weightData.data());

    runtime->run(g);

    auto output = op->getOutput(0);
    auto outShape = output->getShape();

    // Verify output shape: [1, 1, 3, 3]
    ASSERT_EQ(outShape->size(), 4);
    EXPECT_EQ((*outShape)[0]->asConstant(), 1);
    EXPECT_EQ((*outShape)[1]->asConstant(), 1);
    EXPECT_EQ((*outShape)[2]->asConstant(), 3);
    EXPECT_EQ((*outShape)[3]->asConstant(), 3);

    // Copy output data to host
    std::vector<float> outputData(output->getElement());
    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    auto dataBlob = output->getData();
    runtime->memcpy(hostPtr, dataBlob->getRawDataPtr(), output->getTotalBytes(),
                    INFINIRT_MEMCPY_D2H);
    std::memcpy(outputData.data(), hostPtr, output->getTotalBytes());
    runtime->deallocHost(hostPtr);

    // Expected output for 2x2 kernel with all ones:
    // Input:  1  2  3  4    Output: 14 18 22
    //         5  6  7  8            30 34 38
    //         9 10 11 12            46 50 54
    //        13 14 15 16
    std::vector<float> expectedOutput = {14, 18, 22, 30, 34, 38, 46, 50, 54};

    ASSERT_EQ(outputData.size(), expectedOutput.size());
    for (size_t i = 0; i < outputData.size(); ++i) {
        EXPECT_FLOAT_EQ(outputData[i], expectedOutput[i])
            << "Mismatch at index " << i;
    }

    std::cout << "CPU Conv (NoBias) Test PASSED" << std::endl;
}

#ifdef USE_CUDA
// Single device test - NVIDIA F32
TEST(Conv, Conv_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {1, 3, 5, 5};
    Shape shapeWeight = {2, 3, 3, 3};
    Shape shapeBias = {2};
    vector<size_t> strides = {1, 1};
    vector<size_t> dilations = {1, 1};
    vector<size_t> paddings = {1, 1};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto weight = g->addTensor(shapeWeight, DataType(INFINI_DTYPE_F32));
    auto bias = g->addTensor(shapeBias, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ConvObj>(input, weight, bias, strides, dilations,
                                paddings, nullptr);

    // Set input data
    std::vector<float> inputData(input->getElement());
    std::vector<float> weightData(weight->getElement());
    std::vector<float> biasData(bias->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = 1.0f;
    }
    for (size_t i = 0; i < weightData.size(); ++i) {
        weightData[i] = 0.1f;
    }
    for (size_t i = 0; i < biasData.size(); ++i) {
        biasData[i] = 0.5f;
    }

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(inputData.data());
    weight->setData(weightData.data());
    bias->setData(biasData.data());
    // Allocate memory (output only, inputs are externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    void *deviceWeight = runtime->allocDevice(weight->getTotalBytes());
    void *deviceBias = runtime->allocDevice(bias->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    runtime->memcpy(deviceWeight, weightData.data(), weight->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    runtime->memcpy(deviceBias, biasData.data(), bias->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);
    weight->setData(deviceWeight);
    bias->setData(deviceBias);

    // Execute computation
    runtime->run(g);

    // Get output and verify
    auto output = op->getOutput(0);
    auto outShape = output->getShape();

    // Verify output shape: [1, 2, 5, 5]
    ASSERT_EQ(outShape->size(), 4);
    EXPECT_EQ((*outShape)[0]->asConstant(), 1);
    EXPECT_EQ((*outShape)[1]->asConstant(), 2);
    EXPECT_EQ((*outShape)[2]->asConstant(), 5);
    EXPECT_EQ((*outShape)[3]->asConstant(), 5);

    // Copy output data to host
    std::vector<float> outputData(output->getElement());
    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    auto dataBlob = output->getData();
    runtime->memcpy(hostPtr, dataBlob->getRawDataPtr(), output->getTotalBytes(),
                    INFINIRT_MEMCPY_D2H);
    std::memcpy(outputData.data(), hostPtr, output->getTotalBytes());
    runtime->deallocHost(hostPtr);

    // Verify output values are non-zero (sanity check)
    for (size_t i = 0; i < outputData.size(); ++i) {
        EXPECT_GT(outputData[i], 0.0f)
            << "Output at index " << i << " is zero or negative";
    }

    std::cout << "NVIDIA F32 Conv Test PASSED" << std::endl;

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    runtime->deallocDevice(deviceWeight);
    runtime->deallocDevice(deviceBias);
    // Clean up output memory allocated by dataMalloc
    auto outputDataPtr = output->getData();
    if (outputDataPtr) {
        runtime->deallocDevice(outputDataPtr->getRawDataPtr());
    }
}
#endif

} // namespace infini
