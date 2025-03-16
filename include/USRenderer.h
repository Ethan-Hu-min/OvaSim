#include "CUDABuffer.h"
#include "USLaunchParams.h"
#include "Scene.h"
#include <GLFW/glfw3.h>
#include <gl/GL.h>

//
//
//

struct HardwareInfo {
	vec3f pos;
	vec3f angle;
	bool needleSwitch;
	float needleDepth;
};

class USRenderer {
public:
	USRenderer(const Scene* scene);
	~USRenderer();
	void render();
	void resize(const vec2i& newSize);

	void paraResize(int w, int h);

	void postProcess();
	void downloadPixels();
	void setTransducer(const Transducer& transducer);
	void getTransducer();
	void changeTransducer(float angle, const vec3f& axis);
	void changeTransducerAbs(float angleRoll, float anglePitch, float angleYaw);
	void setNeedle(float r_angle, float r_depth);
	void changeNeedle(float changeDepth);
	void changeNeedleAbs(float changeDepth);


	void getCollideId();
	int    getCollideModel();


	void downloadCollideInfo();

	uint32_t* pixelsData();

	void run();
	void clear();
	void initTexture();
	float getNeedleEndX(float x);
	float getNeedleEndY(float y);
	int getNowObtainedNums();

	void updateAccel();

protected:
	void initOptix();
	void createContext();
	void createModule();
	void createRaygenPrograms();
	void createMissPrograms();
	void createHitgroupPrograms();
	void createPipeline();
	void buildSBT();
	void buildAccel();

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

	OptixTraversableHandle asHandle{ 0 };
	std::vector<OptixBuildInput> triangleInput;
	CUDABuffer tempBuffer;
	CUDABuffer outputBuffer;
	OptixAccelEmitDesc emitDesc;
	OptixAccelBuildOptions accelOptions = {};
	OptixAccelBuildOptions accelOptionsUpdate = {};

	USLaunchParams uslaunchParams;
	vec3f originPos;
	vec3f originDir;
	vec3f originVer;
	vec3f originHor;

	CUDABuffer launchParamsBuffer;
	CUDABuffer colorBuffer;
	CUDABuffer intensityBuffer;
	std::vector<CUDABuffer> textureBuffer;
	CUDABuffer noiseBuffer;
	vec2i noiseSize;

	CUDABuffer postprocessBuffer;
	Transducer lastSetTransducer;
	CUDABuffer collide_models_id;
	CUDABuffer collide_models_pos;

	const Scene* scene = nullptr;
	std::vector<CUDABuffer> vertexBuffer;
	std::vector<CUDABuffer> indexBuffer;
	std::vector<CUDABuffer> normalBuffer;

	std::vector<CUDABuffer> originVertexBuffer;

	CUDABuffer asBuffer;
	std::vector<CUdeviceptr> d_vertices;
	std::vector<CUdeviceptr> d_indices;

	std::vector<CUdeviceptr> origin_vertices;
	std::vector<uint32_t> triangleInputFlags;

	std::vector<uint32_t> pixels;
	std::vector<uint8_t> collide_id;
	std::vector<vec3f> collide_pos;
	std::unordered_map<int, float> ovam_scale;
	std::string examplePath = "";

	HardwareInfo renderHWinfo;

	int numMeshes = 0;

	int now_collide_model = -1;
	int now_collide_ovam = -1;
	vec3f now_ovam_pos = vec3f(0.0, 0.0, 0.0);

};