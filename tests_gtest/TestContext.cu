#include "Utils.h"

using namespace cuMat;

TEST(ContextTest, DefaultContextCreation)
{
    Context& ctx = Context::current();
    EXPECT_NE(nullptr, ctx.stream());
}

TEST(ContextTest, MultipleContextInstances)
{
    Context& ctx1 = Context::current();
    Context& ctx2 = Context::current();
    EXPECT_EQ(&ctx1, &ctx2);
}

TEST(ContextTest, MemoryAllocationAndDeallocation)
{
    Context& ctx = Context::current();
    void* ptr = ctx.mallocDevice(1024);
    ASSERT_NE(nullptr, ptr);
    ctx.freeDevice(ptr);
}

TEST(ContextTest, HostMemoryAllocation)
{
    Context& ctx = Context::current();
    void* ptr = ctx.mallocHost(256);
    ASSERT_NE(nullptr, ptr);
    ctx.freeHost(ptr);
}

TEST(ContextTest, HostDeviceTransfer)
{
    const int n = 256;
    std::vector<float> host_src(n, 42.0f);
    std::vector<float> host_dst(n, 0.0f);
    float* d_ptr = nullptr;

    CUMAT_SAFE_CALL(cudaMalloc(&d_ptr, n * sizeof(float)));
    CUMAT_SAFE_CALL(cudaMemcpy(d_ptr, host_src.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUMAT_SAFE_CALL(cudaMemcpy(host_dst.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUMAT_SAFE_CALL(cudaFree(d_ptr));

    for (int i = 0; i < n; ++i)
        EXPECT_FLOAT_EQ(42.0f, host_dst[i]);
}

TEST(ContextTest, StreamSync)
{
    Context& ctx = Context::current();
    CUMAT_SAFE_CALL(cudaStreamSynchronize(ctx.stream()));
    SUCCEED();
}

TEST(ContextTest, FreeDeviceMemory)
{
    size_t freeMem = Context::getFreeDeviceMemory();
    EXPECT_GT(freeMem, 0ULL);
}
