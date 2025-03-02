#pragma once

#include<stdint.h>

#include <cuda_runtime.h>

#include <cuda.h>

extern "C" void postProcess_gpu(int _SampleNum,float* _SrcIntensity, uint32_t* _SrcImg, uint32_t* _DstImg, int _rows, int cols, int _kernal_size_x, int _kernal_size_y, CUstream _stream);

