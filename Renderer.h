#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Scene.h"

struct Camera {
	vec3f from;
	vec3f at;
	vec3f up;
};

class Renderer {
public:
	Renderer(const Scene* scene);
	void render();
	void resize(const vec2i& newSize);
	void downloadPixels(uint32_t h_pixels[]);
	void setCamera(const Camera& camera);
	void getCamera();


protected:
	void initOptix();
	void createContext();
	void createModule();
	void createRaygenPrograms();
	void createMissPrograms();
	void createHitgroupPrograms();
	void createPipeline();
	void buildSBT();
	OptixTraversableHandle buildAccel();
protected:
	CUcontext cudaContext;
	CUstream  stream;
	cudaDeviceProp deviceProps;
	OptixDeviceContextOptions contextOptions = {};
	OptixDeviceContext optixContext;

	OptixPipeline pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions    pipelineLinkOptions = {};

	OptixModule module = nullptr;
	OptixModuleCompileOptions moduleCompileOptions = {};

	std::vector<OptixProgramGroup> raygenPGs;
	CUDABuffer raygenRecordsBuffer;
	std::vector<OptixProgramGroup> missPGs;
	CUDABuffer missRecordsBuffer;
	std::vector<OptixProgramGroup> hitgroupPGs;
	CUDABuffer hitgroupRecordsBuffer;
	OptixShaderBindingTable sbt = {};

	LaunchParams launchParams;
	CUDABuffer launchParamsBuffer;

	CUDABuffer colorBuffer;
	Camera lastSetCamera;

	const Scene* scene;
	std::vector<CUDABuffer> vertexBuffer;
	std::vector<CUDABuffer> indexBuffer;
	std::vector<CUDABuffer> normalBuffer;
	CUDABuffer asBuffer;

};