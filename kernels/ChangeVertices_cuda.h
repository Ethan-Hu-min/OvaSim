#pragma once
#include <stdio.h>
#include<stdint.h>

#include <cuda_runtime.h>

#include <cuda.h>

#include <sutil/vec_math.h>



extern "C" __host__ void changeVerticesPos(float3* outVertices, float3* originVertices, float3 changeModelCenter, int verticesSize, float modelScale);