#include <stdio.h>

#include "ChangeVertices_cuda.h"




extern "C" __global__ void change_vertices(float3* outVertices, float3* originVertices, float3 changeModelCenter, int verticesSize, float modelScale, int width, int height) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//const float scaleValue = abs(cos(changeTime)) * 0.6;

	if (idx < verticesSize) {
		const float3 originVertice = originVertices[idx];
		const float3 changeDir = changeModelCenter - originVertice;
		const float3 changeVertice = outVertices[idx];
		outVertices[idx] = originVertice + changeDir * (1 - modelScale);

		//if (idx == 100000) {
		//	printf("idx: %d , vertice: %f %f %f \n", idx, changeVertice.x, changeVertice.y, changeVertice.z);
		//	printf("time: %f,scale: %f, changeDir: %f %f %f \n", changeTime, scaleValue, changeDir.x, changeDir.y, changeDir.z);
		//}
	}
}


extern "C" __host__ void changeVerticesPos(float3* outVertices, float3* originVertices, float3 changeModelCenter, int verticesSize, float modelScale) {

	const int blockMaxSize = 512;
	int threadSize = verticesSize % blockMaxSize == 0 ? verticesSize / blockMaxSize : (verticesSize / blockMaxSize + 1);
	dim3 threadsPerBlock(threadSize, 1);
	change_vertices << <blockMaxSize, threadsPerBlock >> > (outVertices, originVertices, changeModelCenter, verticesSize, modelScale, blockMaxSize, threadSize);
}