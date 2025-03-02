#pragma once
#include <stdio.h>
#include<stdint.h>

#include <cuda_runtime.h>

#include <cuda.h>

extern "C" __host__ void createTexture(int var, int mean, int textureSize, uint8_t* texture);