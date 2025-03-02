#include <stdio.h>

#include "ImageProcess_cuda.h"

#include <device_launch_parameters.h>

static  __forceinline__ __device__
uint32_t calculate_color(float attenuation) {
    int intensity = static_cast<int>(attenuation * 255);
    if (intensity < 0)intensity = 0;
    if (intensity > 255)intensity = 255;
    return 0xff000000 | (intensity << 0) | (intensity << 8) | (intensity << 16);
}

__global__ void inter2color(int _SampleNum,  float* _SrcInt, uint32_t* _DstImg, int _rows, int _cols, int _kernal_size_x) {
    int _row_idx = threadIdx.x + blockIdx.x * blockDim.x;
    //行坐标
    int _col_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if ((_row_idx < _kernal_size_x) || _row_idx > (_rows - _kernal_size_x) || _col_idx > _cols) {
        return;
    }
    int _img_idx = _row_idx + _col_idx * _rows;
    float nowintensity = _SrcInt[_img_idx];
    _DstImg[_img_idx] = calculate_color(nowintensity / float(_SampleNum));
}


__global__ void postProcess_x( uint32_t* _SrcImg, uint32_t* _DstImg, int _rows, int _cols, int _kernal_size_x) {
    //列坐标
    int _row_idx = threadIdx.x + blockIdx.x * blockDim.x;
    //行坐标
    int _col_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if ((_row_idx < _kernal_size_x) || _row_idx > (_rows - _kernal_size_x) || _col_idx > _cols) {
        return;
    }
    int _img_idx = _row_idx + _col_idx * _rows;

    uint8_t  src_r = ((_SrcImg[_img_idx] >> 16) & 0xFF) / 2;
    uint8_t  src_g = ((_SrcImg[_img_idx] >> 8) & 0xFF) / 2;
    uint8_t  src_b = ((_SrcImg[_img_idx] >> 0) & 0xFF) / 2;

    for (int offset = 1; offset <= _kernal_size_x / 2; offset++) {
        int _src_idx_r = _row_idx + offset + _col_idx * _rows;
        int _src_idx_l = _row_idx - offset + _col_idx * _rows;
        src_r += (0.5 / float(_kernal_size_x - 1)) * ((_SrcImg[_src_idx_r] >> 16) & 0xFF);
        src_r += (0.5 / float(_kernal_size_x - 1)) * ((_SrcImg[_src_idx_l] >> 16) & 0xFF);
        src_g += (0.5 / float(_kernal_size_x - 1)) * ((_SrcImg[_src_idx_r] >> 8) & 0xFF);
        src_g += (0.5 / float(_kernal_size_x - 1)) * ((_SrcImg[_src_idx_l] >> 8) & 0xFF);
        src_b += (0.5 / float(_kernal_size_x - 1)) * ((_SrcImg[_src_idx_r] >> 0) & 0xFF);
        src_b += (0.5 / float(_kernal_size_x - 1)) * ((_SrcImg[_src_idx_l] >> 0) & 0xFF);
    }
    uint32_t dst_color = 0xff000000
        | (src_b << 0) | (src_g << 8) | (src_r << 16);

    _DstImg[_img_idx] = dst_color;

}

__global__ void postProcess_y(uint32_t* _SrcImg, uint32_t* _DstImg, int _rows, int _cols, int _kernal_size_y) {
    //列坐标
    int _row_idx = threadIdx.x + blockIdx.x * blockDim.x;
    //行坐标
    int _col_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if (_row_idx > _rows || _col_idx > _cols) {
        return;
    }
    int _img_idx = _row_idx + _col_idx * _rows;
    uint8_t  src_r = ((_DstImg[_img_idx] >> 16) & 0xFF) / 2;
    uint8_t  src_g = ((_DstImg[_img_idx] >> 8) & 0xFF) / 2;
    uint8_t  src_b = ((_DstImg[_img_idx] >> 0) & 0xFF) / 2;

    for (int offset = 1; offset <= _kernal_size_y / 2; offset++) {
        int _src_idx_r = _row_idx + offset + _col_idx * _rows;
        int _src_idx_l = _row_idx - offset + _col_idx * _rows;
        src_r += (0.5 / float(_kernal_size_y - 1)) * ((_DstImg[_src_idx_r] >> 16) & 0xFF);
        src_r += (0.5 / float(_kernal_size_y - 1)) * ((_DstImg[_src_idx_l] >> 16) & 0xFF);
        src_g += (0.5 / float(_kernal_size_y - 1)) * ((_DstImg[_src_idx_r] >> 8) & 0xFF);
        src_g += (0.5 / float(_kernal_size_y - 1)) * ((_DstImg[_src_idx_l] >> 8) & 0xFF);
        src_b += (0.5 / float(_kernal_size_y - 1)) * ((_DstImg[_src_idx_r] >> 0) & 0xFF);
        src_b += (0.5 / float(_kernal_size_y - 1)) * ((_DstImg[_src_idx_l] >> 0) & 0xFF);
    }
    uint32_t dst_color = 0xff000000
        | (src_b << 0) | (src_g << 8) | (src_r << 16);

    _SrcImg[_img_idx] = dst_color;

}

__global__ void postProcess_mask(uint32_t* _SrcImg, int _rows, int _cols) {
    //列坐标
    int _row_idx = threadIdx.x + blockIdx.x * blockDim.x;
    //行坐标
    int _col_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if (_row_idx > _rows || _col_idx > _cols) {
        return;
    }
    uint32_t dst_color = 0xff000000;
    int _img_idx = _row_idx + _col_idx * _rows;;
    int dis = (_row_idx - _rows / 2) * (_row_idx - _rows / 2) + _col_idx * _col_idx;
    if (dis < 625 || dis > _cols * _cols) {
        _SrcImg[_img_idx] = dst_color;
    }
    if(float(_row_idx) *  0.8 * (-float(_cols) / float(_rows)) + float(_cols) * 0.4  > float(_col_idx))_SrcImg[_img_idx] = dst_color;
    if(float(_row_idx) *  0.8 * (float(_cols) / float(_rows)) - float(_cols) * 0.4  > float(_col_idx))_SrcImg[_img_idx] = dst_color;
}


extern "C" __host__ void postProcess_gpu(int _SampleNum, float* _SrcIntensity,  uint32_t* _SrcImg, uint32_t* _DstImg, int _rows, int _cols, int _kernal_size_x, int _kernal_size_y, CUstream _stream) {

    dim3 block_size(32, 32);
    int row_size = _rows % block_size.x == 0 ? _rows / block_size.x : (_rows / block_size.x + 1);
    int col_size = _cols % block_size.y == 0 ? _cols / block_size.y : (_cols / block_size.y + 1);
    dim3 thread_size(row_size, col_size);

    inter2color << < block_size, thread_size, 0, _stream >> > (_SampleNum, _SrcIntensity, _SrcImg,  _rows, _cols, _kernal_size_x);
    postProcess_x <<< block_size, thread_size, 0, _stream >>> (_SrcImg, _DstImg, _rows, _cols, _kernal_size_x);
    postProcess_y <<< block_size, thread_size, 0, _stream >>> (_SrcImg, _DstImg, _rows, _cols, _kernal_size_y);
    postProcess_mask << <block_size, thread_size, 0, _stream >> > (_SrcImg, _rows, _cols);
}
