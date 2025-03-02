#include "CreateTexture_cuda.h"

#include <curand_kernel.h>

// CUDA �˺������� texture ���������ݸ�ֵΪ���� VAR, MEAN �����ĸ�˹�ֲ������
__global__ void create_Texture(int var, int mean, uint8_t* texture, curandState* states) {
    // ��ȡ�̵߳�ȫ������
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ��ʼ�� CURAND ״̬
    curand_init(clock64(), idx, 0, &states[idx]);

    // ���ɷ��ϸ�����ֵ�ͷ���ĸ�˹�ֲ������
    float rand_num = curand_normal(&states[idx]) * (float)var + (float)mean;
    // �������ת��Ϊ uint8_t ���ͣ�����ֵ�� texture
    texture[idx] = (uint8_t)rand_num;


    //// ���� [0, 1) ��Χ�ڵľ��ȷֲ����������
    //float rand_num = curand_uniform(&states[idx]);

    //// �����������ӳ�䵽 [0, 255] ��Χ��ת��Ϊ uint8_t ����
    //texture[idx] = static_cast<uint8_t>(rand_num * 255);

}

// �������������� CUDA �˺���
void createTexture(int var, int mean, int textureSize, uint8_t* texture) {
    // ������С
    const int blockSize = 256;
    // ���������С
    const int gridSize = (textureSize + blockSize - 1) / blockSize;

    // ���� CURAND ״̬������豸�ڴ�
    curandState* d_states;
    cudaMalloc((void**)&d_states, textureSize * sizeof(curandState));

    // ���� CUDA �˺���
    create_Texture << <gridSize, blockSize >> > (var, mean, texture, d_states);

    // ͬ���豸
    cudaDeviceSynchronize();

    // �ͷ� CURAND ״̬������豸�ڴ�
    cudaFree(d_states);
}
