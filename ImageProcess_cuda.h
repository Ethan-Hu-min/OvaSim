#pragma once

#include<stdint.h>

#include <cuda_runtime.h>

#include <cuda.h>
void postProcess_gpu(uint32_t* _SrcImg, uint32_t* _DstImg, int _rows, int cols, int _kernal_size_x, int _kernal_size_y, CUstream _stream);