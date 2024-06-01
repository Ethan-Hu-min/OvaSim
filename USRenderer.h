#include "CUDABuffer.h"
#include "USLaunchParams.h"
#include "Scene.h"
#include <GLFW/glfw3.h>
#include <gl/GL.h>

class USRenderer {
public:
	USRenderer(const Scene* scene);
	void render();
	void resize(const vec2i& newSize);
	void postProcess();
	void downloadPixels(uint32_t h_pixels[]);
	void setTransducer(const Transducer& transducer);
	void getTransducer();
	void loadTexture(std::vector<std::string>& filmname);
	void changeTransducer(float angle, const vec3f& axis);
	void run();
	void clear();


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

	USLaunchParams uslaunchParams;
	CUDABuffer launchParamsBuffer;
	CUDABuffer colorBuffer;
	CUDABuffer postprocessBuffer;
	Transducer lastSetTransducer;

	const Scene* scene;
	std::vector<CUDABuffer> vertexBuffer;
	std::vector<CUDABuffer> indexBuffer;
	std::vector<CUDABuffer> normalBuffer;
	CUDABuffer asBuffer;

	std::vector<uint32_t> pixels;
	
	std::string examplePath;
};