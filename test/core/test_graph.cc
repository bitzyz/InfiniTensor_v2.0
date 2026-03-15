#include "core/graph.h"
#include "core/runtime.h"
#include "operators/Gemm.h"
#include "operators/Unary.h"
#include "gtest/gtest.h"

namespace infini {
class GraphBasicTest : public testing::Test {
  protected:
    Runtime runtime;

    void SetUp() override { runtime = make_ref<RuntimeObj>(); }
};

// Test Graph construction and runtime retrieval
TEST_F(GraphBasicTest, GraphConstruction) {
    auto graph = make_ref<GraphObj>(runtime);

    EXPECT_EQ(graph->getRuntime(), runtime);
    EXPECT_TRUE(graph->getTensors().empty());
    EXPECT_TRUE(graph->getOperators().empty());
    EXPECT_TRUE(graph->getOutputs().empty());
}

// Test adding constant shape Tensor
TEST_F(GraphBasicTest, AddConcreteTensor) {
    auto graph = make_ref<GraphObj>(runtime);

    auto tensor1 = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto tensor2 = graph->addTensor({5, 6}, DataType(INFINI_DTYPE_F16));

    EXPECT_EQ(graph->getTensors().size(), 2);
    EXPECT_EQ(tensor1->getRank(), 3);
    EXPECT_EQ(tensor2->getRank(), 2);
    EXPECT_EQ(tensor1->getDataType(), DataType(INFINI_DTYPE_F32));
    EXPECT_EQ(tensor2->getDataType(), DataType(INFINI_DTYPE_F16));
}

// Test adding symbolic shape Tensor
TEST_F(GraphBasicTest, AddSymbolicTensor) {
    auto graph = make_ref<GraphObj>(runtime);

    auto batch = ExprObj::variable("batch");
    auto height = ExprObj::constant(224);
    auto width = ExprObj::constant(224);

    auto shapeExpr = ShapeExpr(new ShapeExprObj({batch, height, width}));
    auto tensor = graph->addTensor(shapeExpr, DataType(INFINI_DTYPE_F32));

    EXPECT_EQ(graph->getTensors().size(), 1);
    EXPECT_EQ(tensor->getRank(), 3);
    EXPECT_FALSE(tensor->getShape()->isConcrete());
    EXPECT_TRUE(tensor->getShape()->isDynamic());
}

// Test adding Tensor with stride
TEST_F(GraphBasicTest, AddTensorWithStride) {
    auto graph = make_ref<GraphObj>(runtime);

    // Constant stride
    auto tensor1 =
        graph->addTensor({2, 3, 4}, {12, 4, 1}, DataType(INFINI_DTYPE_F32));
    EXPECT_TRUE(tensor1->getStride()->isConcrete());

    // Stride expression
    auto strideExpr = StrideExpr(new StrideExprObj(
        {ExprObj::constant(12), ExprObj::constant(4), ExprObj::constant(1)}));
    auto tensor2 =
        graph->addTensor({2, 3, 4}, strideExpr, DataType(INFINI_DTYPE_F32));
    EXPECT_TRUE(tensor2->getStride()->isConcrete());
}

// Test batch adding Tensors
TEST_F(GraphBasicTest, AddTensorVector) {
    auto graph = make_ref<GraphObj>(runtime);

    TensorVec tensors = {
        make_ref<TensorObj>(Shape{2, 3}, DataType(INFINI_DTYPE_F32)),
        make_ref<TensorObj>(Shape{4, 5}, DataType(INFINI_DTYPE_F16)),
        make_ref<TensorObj>(Shape{6, 7, 8}, DataType(INFINI_DTYPE_F32))};

    auto added = graph->addTensor(tensors);

    EXPECT_EQ(graph->getTensors().size(), 3);
    EXPECT_EQ(added.size(), 3);
    EXPECT_EQ(added[0], tensors[0]);
    EXPECT_EQ(added[1], tensors[1]);
    EXPECT_EQ(added[2], tensors[2]);
}

// Test removing Tensor and Operator
TEST_F(GraphBasicTest, RemoveTensorAndOperator) {
    auto graph = make_ref<GraphObj>(runtime);

    // Create Tensor and Operator
    auto A = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({3, 4}, DataType(INFINI_DTYPE_F32));
    auto Y = graph->addTensor({1, 2, 4}, DataType(INFINI_DTYPE_F32));
    auto gemm = graph->addOpWithOutputs<GemmObj>(A, B, Y, nullptr);

    EXPECT_EQ(graph->getTensors().size(), 3);
    EXPECT_EQ(graph->getOperators().size(), 1);

    // Remove Tensor
    graph->removeTensor(A);
    EXPECT_EQ(graph->getTensors().size(), 2);

    // Remove Operator
    graph->removeOperator(gemm);
    EXPECT_EQ(graph->getOperators().size(), 0);
}

TEST_F(GraphBasicTest, SetOutputs) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));

    graph->setOutputs({input});

    ASSERT_EQ(graph->getOutputs().size(), 1);
    EXPECT_EQ(graph->getOutputs()[0], input);
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphBasicTest, CudaGraphLifecycleBookkeeping) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    graph->setOutputs({input});

    EXPECT_FALSE(graph->isCudaGraphEnabled());
    auto invalidateBase = graph->getCudaGraphInvalidateCount();
    graph->setCudaGraphEnabled(true);
    EXPECT_TRUE(graph->isCudaGraphEnabled());
    EXPECT_EQ(graph->getCudaGraphInvalidateCount(), invalidateBase + 1);

    EXPECT_FALSE(graph->hasCompiledCudaGraphForCurrentThread());
    auto compileBase = graph->getCudaGraphCompileCount();
    graph->markCudaGraphCompiledForCurrentThread();
    EXPECT_TRUE(graph->hasCompiledCudaGraphForCurrentThread());
    EXPECT_EQ(graph->getCudaGraphCompileCount(), compileBase + 1);

    auto launchBase = graph->getCudaGraphLaunchCount();
    graph->markCudaGraphLaunched();
    EXPECT_EQ(graph->getCudaGraphLaunchCount(), launchBase + 1);

    invalidateBase = graph->getCudaGraphInvalidateCount();
    graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    EXPECT_FALSE(graph->hasCompiledCudaGraphForCurrentThread());
    EXPECT_EQ(graph->getCudaGraphInvalidateCount(), invalidateBase + 1);
}

TEST_F(GraphBasicTest, CudaGraphInvalidatesOnOutputMutation) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    auto output = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));

    graph->setOutputs({input});
    graph->markCudaGraphCompiledForCurrentThread();
    EXPECT_TRUE(graph->hasCompiledCudaGraphForCurrentThread());

    auto invalidateBase = graph->getCudaGraphInvalidateCount();
    graph->setOutputs({output});
    EXPECT_FALSE(graph->hasCompiledCudaGraphForCurrentThread());
    EXPECT_EQ(graph->getCudaGraphInvalidateCount(), invalidateBase + 1);
}

#ifdef USE_CUDA
TEST_F(GraphBasicTest, CudaGraphCompileLaunchAndRecompileAfterMutation) {
    RuntimeObj::init();
    int nvidiaCount = 0;
    CHECK_INFINI_ERROR(
        infinirtGetDeviceCount(INFINI_DEVICE_NVIDIA, &nvidiaCount));
    if (nvidiaCount <= 0) {
        GTEST_SKIP() << "No NVIDIA device available for CUDA graph test";
    }

    runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    auto graph = make_ref<GraphObj>(runtime);
    graph->setCudaGraphEnabled(true);

    auto x = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    auto relu = graph->addOp<UnaryObj>(OpType::Relu, x, nullptr);
    graph->setOutputs({relu->getOutput(0)});

    std::vector<float> input = {-1.0f, 0.5f, -2.0f, 3.0f};
    x->setData(input.data());
    runtime->dataMalloc(graph);

    auto compileBase = graph->getCudaGraphCompileCount();
    auto launchBase = graph->getCudaGraphLaunchCount();

    runtime->run(graph); // compile path
    EXPECT_EQ(graph->getCudaGraphCompileCount(), compileBase + 1);
    EXPECT_EQ(graph->getCudaGraphLaunchCount(), launchBase);

    runtime->run(graph); // launch cached graph
    EXPECT_EQ(graph->getCudaGraphCompileCount(), compileBase + 1);
    EXPECT_EQ(graph->getCudaGraphLaunchCount(), launchBase + 1);

    auto invalidateBase = graph->getCudaGraphInvalidateCount();
    graph->addTensor({1}, DataType(INFINI_DTYPE_F32)); // force invalidation
    EXPECT_EQ(graph->getCudaGraphInvalidateCount(), invalidateBase + 1);

    runtime->run(graph); // recompile after mutation
    EXPECT_EQ(graph->getCudaGraphCompileCount(), compileBase + 2);

    runtime->run(graph); // relaunch new cached graph
    EXPECT_EQ(graph->getCudaGraphLaunchCount(), launchBase + 2);
}
#endif
} // namespace infini
