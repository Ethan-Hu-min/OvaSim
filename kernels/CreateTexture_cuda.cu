#include "CreateTexture_cuda.h"

#include <curand_kernel.h>

// CUDA 核函数：将 texture 的所有数据赋值为符合 VAR, MEAN 参数的高斯分布随机数
__global__ void create_Texture(int var, int mean, uint8_t* texture, curandState* states) {
    // 获取线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 初始化 CURAND 状态
    curand_init(clock64(), idx, 0, &states[idx]);

    // 生成符合给定均值和方差的高斯分布随机数
    float rand_num = curand_normal(&states[idx]) * (float)var + (float)mean;
    // 将随机数转换为 uint8_t 类型，并赋值给 texture
    texture[idx] = (uint8_t)rand_num;


    //// 生成 [0, 1) 范围内的均匀分布随机浮点数
    //float rand_num = curand_uniform(&states[idx]);

    //// 将随机浮点数映射到 [0, 255] 范围并转换为 uint8_t 类型
    //texture[idx] = static_cast<uint8_t>(rand_num * 255);

}

// 主机函数：调用 CUDA 核函数
void createTexture(int var, int mean, int textureSize, uint8_t* texture) {
    // 定义块大小
    const int blockSize = 256;
    // 计算网格大小
    const int gridSize = (textureSize + blockSize - 1) / blockSize;

    // 分配 CURAND 状态数组的设备内存
    curandState* d_states;
    cudaMalloc((void**)&d_states, textureSize * sizeof(curandState));

    // 调用 CUDA 核函数
    create_Texture << <gridSize, blockSize >> > (var, mean, texture, d_states);

    // 同步设备
    cudaDeviceSynchronize();

    // 释放 CURAND 状态数组的设备内存
    cudaFree(d_states);
}
