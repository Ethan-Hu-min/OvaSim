#pragma once

#include<stdint.h>

#include <cuda_runtime.h>

#include <cuda.h>

#include <math.h>

extern "C" void postProcess_gpu(uint8_t* _NoiseImg, int _NoiseRows, int _NoiseCols, const  int _SampleNum, float* _SrcIntensity, uint32_t* _SrcImg, uint32_t* _DstImg, int _rows, int _cols, int _kernal_size_x, int _kernal_size_y, float needleAngle, CUstream _stream);

