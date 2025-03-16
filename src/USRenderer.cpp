#include "USRenderer.h"

#include "cuda.h"
#include <sutil/sutil.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>

#include <iostream>
#include <ImageProcess_cuda.h>

#include <ChangeVertices_cuda.h>
#include <CreateTexture_cuda.h>
#include <GlobalConfig.h>
#include<qdebug.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"



std::ostream& operator<<(std::ostream& os, const vec3f& v) {
    os << v.x << ", " << v.y << ", " << v.z;
    return os;
}

struct  __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* data;
};

struct  __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* data;
};

struct  __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshSBTData data;
};


USRenderer::USRenderer(const Scene* scene) :scene(scene) {
    initOptix();
    qDebug() << "#creating optix context ..." ;
    createContext();

    qDebug() << "#:setting up module ..." ;
    createModule();

    qDebug() << "#creating raygen programs ..." ;
    createRaygenPrograms();
    qDebug() << "#creating miss programs ..." ;
    createMissPrograms();
    qDebug() << "#creating hitgroup programs ..." ;
    createHitgroupPrograms();
    buildAccel();
    uslaunchParams.traversable = asHandle;
    qDebug() << " handle:" << int(uslaunchParams.traversable) ;

    qDebug() << "#setting up optix pipeline ..." ;
    createPipeline();

    qDebug() << "#building SBT ..." ;
    buildSBT();
    launchParamsBuffer.alloc(sizeof(uslaunchParams));
    qDebug() << "sizeInBytes: " << launchParamsBuffer.sizeInBytes ;
    qDebug() << "launchParams: " << sizeof(uslaunchParams) ;
    qDebug() << "#context, module, pipeline, etc, all set up ..." ;
    qDebug() << GDT_TERMINAL_GREEN;
    qDebug() << "#Scene fully set up" ;
    qDebug() << GDT_TERMINAL_DEFAULT;
    for (int i = 0; i < scene->models.size(); i++) {
        ovam_scale.insert(std::make_pair(i, 1.0));
    }
}

USRenderer::~USRenderer() {
    if (scene != nullptr) delete scene;

    qDebug() << "USRenderer delete";


}

void USRenderer::initOptix() {
    qDebug() << "#initializing optix..." ;
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("#osc: no CUDA capable devices found!");
    qDebug() << "#osc: found " << numDevices << " CUDA devices" ;

    OPTIX_CHECK(optixInit());
    qDebug() << GDT_TERMINAL_GREEN
        << "#osc: successfully initialized optix... yay!"
        << GDT_TERMINAL_DEFAULT ;
}

void USRenderer::initTexture() {
    int n = 7;
    int _width = uslaunchParams.frame.size.x;
    int _height = uslaunchParams.frame.size.y;
    textureBuffer.resize(n);
    for (int i = 0; i < n; i++) {
        textureBuffer[i].resize(_width * _height  * sizeof(uint8_t));
    }
    uslaunchParams.textures.bgTexture = (uint8_t*)textureBuffer[0].d_pointer();
    uslaunchParams.textures.bladderTexture = (uint8_t*)textureBuffer[1].d_pointer();
    uslaunchParams.textures.uterusTexture = (uint8_t*)textureBuffer[2].d_pointer();
    uslaunchParams.textures.uterusinTexture = (uint8_t*)textureBuffer[3].d_pointer();
    uslaunchParams.textures.intestineTexture = (uint8_t*)textureBuffer[4].d_pointer();
    uslaunchParams.textures.ovaryTexture = (uint8_t*)textureBuffer[5].d_pointer();
    uslaunchParams.textures.ovamTexture = (uint8_t*)textureBuffer[6].d_pointer();

    createTexture(26, 93, _width * _height, (uint8_t*)textureBuffer[0].d_pointer());
    createTexture(15, 37, _width * _height, (uint8_t*)textureBuffer[1].d_pointer());
    createTexture(23, 103, _width * _height, (uint8_t*)textureBuffer[2].d_pointer());
    createTexture(24, 125, _width * _height, (uint8_t*)textureBuffer[3].d_pointer());
    createTexture(100, 110, _width * _height, (uint8_t*)textureBuffer[4].d_pointer());
    createTexture(34, 90, _width * _height, (uint8_t*)textureBuffer[5].d_pointer());
    createTexture(11, 26, _width * _height, (uint8_t*)textureBuffer[6].d_pointer());

    int berlinWidth, berlinHeight, channels;
    unsigned char* host_data = stbi_load((GlobalConfig::dataPath + GlobalConfig::berlinNoisePath).c_str(), &berlinWidth, &berlinHeight, &channels, 1);
    if (host_data) {
        noiseBuffer.alloc(berlinWidth * berlinHeight * sizeof(uint8_t));
        noiseBuffer.upload(host_data, berlinWidth * berlinHeight * sizeof(uint8_t));
        stbi_image_free(host_data);
        noiseSize.x = berlinWidth;
        noiseSize.y = berlinHeight;
        qDebug() << "[INFO] Load Imgae Success";

    }
    else {
        qDebug() << "[ERROR] Failed to load image";
    }
}


static void context_log_cb(unsigned int level,
    const char* tag,
    const char* message,
    void*)
{
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void USRenderer::createContext() {

    const int deviceID = 0;
    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaGetDeviceProperties(&deviceProps, deviceID);
    qDebug() << "#osc: running on device: " << deviceProps.name ;
    //CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    //if (cuRes != CUDA_SUCCESS)
    //    fprintf(stderr, "Error querying current context: error code %d\n", cuRes);
    cudaContext = 0;
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &contextOptions, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
    (optixContext, context_log_cb, nullptr, 4));
}

void USRenderer::createModule() {
    //moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 1;
    size_t      inputSize = 0;
    const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "USdevicePrograms.cu", inputSize);
    //const std::string ptxCode = embedded_ptx_code;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        input,
        inputSize,
        log, &sizeof_log,
        &module
    ));
    if (sizeof_log > 1) PRINT(log);
}

void USRenderer::createRaygenPrograms() {
    raygenPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &raygenPGs[0]
    ));
    if (sizeof_log > 1) PRINT(log);
}

void USRenderer::createMissPrograms() {
    missPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &missPGs[0]
    ));
    if (sizeof_log > 1) PRINT(log);
}

void USRenderer::createHitgroupPrograms() {
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &hitgroupPGs[0]
    ));
    if (sizeof_log > 1) PRINT(log);
}

void USRenderer::createPipeline() {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(optixContext,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups.data(),
        (int)programGroups.size(),
        log, &sizeof_log,
        &pipeline
    ));
    if (sizeof_log > 1) PRINT(log);

    OPTIX_CHECK(optixPipelineSetStackSize
    (/* [in] The pipeline to configure the stack size for */
        pipeline,
        /* [in] The direct stack size requirement for direct
           callables invoked from IS or AH. */
        2 * 1024,
        /* [in] The direct stack size requirement for direct
           callables invoked from RG, MS, or CH.  */
        2 * 1024,
        /* [in] The continuation stack requirement. */
        2 * 1024,
        /* [in] The maximum depth of a traversable graph
           passed to trace. */
        1));
    if (sizeof_log > 1) PRINT(log);
}

void USRenderer::buildAccel() {
    PING;
    numMeshes = scene->worldmodel.size();
    PRINT(numMeshes);
    vertexBuffer.resize(numMeshes);
    indexBuffer.resize(numMeshes);
    normalBuffer.resize(numMeshes);
    originVertexBuffer.resize(numMeshes);
    origin_vertices.resize(numMeshes);

    triangleInput.resize(numMeshes);
    d_vertices.resize(numMeshes);
    d_indices.resize(numMeshes);
    triangleInputFlags.resize(numMeshes);

    for (int meshID = 0; meshID < numMeshes; meshID++) {
        TriangleMesh& mesh = *scene->worldmodel[meshID];
        vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
        originVertexBuffer[meshID].alloc_and_upload(mesh.vertex);

        indexBuffer[meshID].alloc_and_upload(mesh.index);
        if (!mesh.normal.empty())
            normalBuffer[meshID].alloc_and_upload(mesh.normal);

        triangleInput[meshID] = {};
        triangleInput[meshID].type
            = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
        d_indices[meshID] = indexBuffer[meshID].d_pointer();
        origin_vertices[meshID] = originVertexBuffer[meshID].d_pointer();
        triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
        triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
        triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

        triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
        triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
        triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

        triangleInputFlags[meshID] = 0;

        triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
        triangleInput[meshID].triangleArray.numSbtRecords = 1;
        triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }

    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
    (optixContext,
        &accelOptions,
        triangleInput.data(),
        (int)numMeshes,  // num_build_inputs
        &blasBufferSizes
    ));
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));


    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();


    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
    OPTIX_CHECK(optixAccelBuild(
        optixContext,
        0,
        &accelOptions,
        triangleInput.data(),
        (int)numMeshes,
        tempBuffer.d_pointer(),
        tempBuffer.sizeInBytes,

        outputBuffer.d_pointer(),
        outputBuffer.sizeInBytes,

        &asHandle,

        &emitDesc, 1
    ));
    CUDA_SYNC_CHECK();
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    //asBuffer.alloc(compactedSize);
    //OPTIX_CHECK(optixAccelCompact(optixContext,
    //    stream,
    //    asHandle,
    //    asBuffer.d_pointer(),
    //    asBuffer.sizeInBytes,
    //    &asHandle));
    CUDA_SYNC_CHECK();
    tempBuffer.free();
    compactedSizeBuffer.free();
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    CUDA_SYNC_CHECK();
}

int USRenderer::getNowObtainedNums(){
    int nownums = 0;
    for (const auto& pair : ovam_scale) {
        if (pair.second < 0.3)nownums++;
    }
    return nownums;
}


void USRenderer::updateAccel() {
    const int changeModelID = this->now_collide_ovam;
    float changescale = this->ovam_scale[changeModelID];
    if (changescale > 0.05) {
        this->ovam_scale[changeModelID] = changescale - 0.02;
    }
    if (changeModelID > 0) {
       // vec3f now_needle_pos = uslaunchParams.transducer.position + 


        changeVerticesPos((float3*)*triangleInput.data()[changeModelID].triangleArray.vertexBuffers,
            (float3*)origin_vertices[changeModelID],
            make_float3(this->now_ovam_pos.x, this->now_ovam_pos.y, this->now_ovam_pos.z),
            triangleInput.data()[changeModelID].triangleArray.numVertices,
            changescale
        );

        accelOptionsUpdate.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        accelOptionsUpdate.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(
            optixContext,
            0,
            &accelOptionsUpdate,
            triangleInput.data(),
            (int)numMeshes,
            tempBuffer.d_pointer(),
            tempBuffer.sizeInBytes,

            outputBuffer.d_pointer(),
            outputBuffer.sizeInBytes,

            &asHandle,

            nullptr, 0
        ));
        CUDA_SYNC_CHECK();
    }
}

void USRenderer::buildSBT() {
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++) {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = scene->worldmodel.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID = 0; meshID < numObjects; meshID++) {
        auto mesh = scene->worldmodel[meshID];
        HitgroupRecord rec;
        // all meshes use the same code, so all same hit group
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));
        rec.data.color = vec3f(157 / 256, 109 / 256, 253 / 256);
        rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
        rec.data.index = (vec3i*)indexBuffer[meshID].d_pointer();
        rec.data.normal = (vec3f*)normalBuffer[meshID].d_pointer();
        rec.data.indexModelSBT = mesh->indexModel;
        rec.data.materialID = mesh->indexMaterial;
        hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}


void USRenderer::render() {

    //qDebug() << " bug1" ;
    launchParamsBuffer.upload(&uslaunchParams, 1);
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
        pipeline, stream,
        /*! parameters and SBT */
        launchParamsBuffer.d_pointer(),
        launchParamsBuffer.sizeInBytes,
        &sbt,
        /*! dimensions of the launch: */
        uslaunchParams.transducer.nums,
        uslaunchParams.numSamples,
        1
    ));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
}



void USRenderer::setTransducer(const Transducer& _transducer)
{
    lastSetTransducer = _transducer;
    uslaunchParams.transducer.width = _transducer.t_width;
    uslaunchParams.transducer.nums = _transducer.t_nums;
    uslaunchParams.transducer.angle = _transducer.t_angle;
    uslaunchParams.transducer.position = _transducer.t_position;
    uslaunchParams.transducer.direction = normalize(_transducer.t_direction);
    uslaunchParams.transducer.horizontal
        = normalize(cross(uslaunchParams.transducer.direction,
            _transducer.t_vertical));
    uslaunchParams.transducer.vertical
        = normalize(cross(uslaunchParams.transducer.horizontal,
            uslaunchParams.transducer.direction));
    originPos = uslaunchParams.transducer.position;
    originDir = uslaunchParams.transducer.direction;
    originVer = uslaunchParams.transducer.vertical;
    originHor = uslaunchParams.transducer.horizontal;

    uslaunchParams.frame.size.x = _transducer.t_nums;
    uslaunchParams.frame.size.y = _transducer.t_nums;

    uslaunchParams.maxBounce = GlobalConfig::maxBounceNum;

}

void USRenderer::setNeedle(float r_angle, float r_depth) {
    uslaunchParams.needle.relaAngle = r_angle;
    uslaunchParams.needle.relaDepth = r_depth;
}

void USRenderer::changeNeedle(float changeDepth) {
    if ((changeDepth > 0 && uslaunchParams.needle.relaDepth < 1.0f) || (changeDepth < 0 && uslaunchParams.needle.relaDepth > 0.0f)) {
        uslaunchParams.needle.relaDepth += changeDepth;
    }
}

void USRenderer::changeNeedleAbs(float changeDepth) {
    uslaunchParams.needle.relaDepth = changeDepth;
}



void USRenderer::getTransducer() {
    //qDebug() << "C_pos:" << uslaunchParams.transducer.position.x << " " << uslaunchParams.transducer.position.y << " " << uslaunchParams.transducer.position.z ;
    qDebug() << "dir:" << uslaunchParams.transducer.direction[0] << uslaunchParams.transducer.direction[1] << uslaunchParams.transducer.direction[2];
}



vec3f rotateVector(const vec3f& vec, float angle, const vec3f& axis) {
    float rad = angle * M_PI / 180.0; 
    float cosTheta = cos(rad);
    float sinTheta = sin(rad);

    // calcul rotate metrix
    vec3f result;
    result.x = (cosTheta + (1 - cosTheta) * axis.x * axis.x) * vec.x;
    result.x += ((1 - cosTheta) * axis.x * axis.y - axis.z * sinTheta) * vec.y;
    result.x += ((1 - cosTheta) * axis.x * axis.z + axis.y * sinTheta) * vec.z;

    result.y = ((1 - cosTheta) * axis.x * axis.y + axis.z * sinTheta) * vec.x;
    result.y += (cosTheta + (1 - cosTheta) * axis.y * axis.y) * vec.y;
    result.y += ((1 - cosTheta) * axis.y * axis.z - axis.x * sinTheta) * vec.z;

    result.z = ((1 - cosTheta) * axis.x * axis.z - axis.y * sinTheta) * vec.x;
    result.z += ((1 - cosTheta) * axis.y * axis.z + axis.x * sinTheta) * vec.y;
    result.z += (cosTheta + (1 - cosTheta) * axis.z * axis.z) * vec.z;
    return normalize(result);
}

void USRenderer::changeTransducer(float angle, const vec3f& axis) {

    vec3f predir = uslaunchParams.transducer.direction;
    vec3f prever = uslaunchParams.transducer.vertical;
    vec3f prehor = uslaunchParams.transducer.horizontal;
    uslaunchParams.transducer.direction = rotateVector(predir, angle, axis);
    uslaunchParams.transducer.vertical = rotateVector(prever, angle, axis);
    uslaunchParams.transducer.horizontal = rotateVector(prehor, angle, axis);
}

void USRenderer::changeTransducerAbs(float angleRoll, float anglePitch, float angleYaw) {
    vec3f rotateDir1 = rotateVector(originDir, angleRoll, vec3f(1.0f, 0.0f, 0.0f));
    vec3f rotateDir2 = rotateVector(rotateDir1, anglePitch, vec3f(0.0f, 1.0f, 0.0f));
    vec3f rotateDir3 = rotateVector(rotateDir2, angleYaw, vec3f(0.0f, 0.0f, 1.0f));
    vec3f rotateVer1 = rotateVector(originVer, angleRoll, vec3f(1.0f, 0.0f, 0.0f));
    vec3f rotateVer2 = rotateVector(rotateVer1, anglePitch, vec3f(0.0f, 1.0f, 0.0f));
    vec3f rotateVer3 = rotateVector(rotateVer2, angleYaw, vec3f(0.0f, 0.0f, 1.0f));
    vec3f rotateHor1 = rotateVector(originHor, angleRoll, vec3f(1.0f, 0.0f, 0.0f));
    vec3f rotateHor2 = rotateVector(rotateHor1, anglePitch, vec3f(0.0f, 1.0f, 0.0f));
    vec3f rotateHor3 = rotateVector(rotateHor2, angleYaw, vec3f(0.0f, 0.0f, 1.0f));
    uslaunchParams.transducer.direction = normalize(rotateDir3);
    uslaunchParams.transducer.vertical = normalize(rotateVer3);
    uslaunchParams.transducer.horizontal = normalize(rotateHor3);

}


void USRenderer::resize(const vec2i& newSize)
{
    // if window minimized
    if (newSize.x == 0 || newSize.y == 0) return;

    // resize our cuda frame buffer
    colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
    postprocessBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
    intensityBuffer.resize(newSize.x * newSize.y * sizeof(float));

    // update the uslaunch parameters that we'll pass to the optix
    // uslaunch:
    uslaunchParams.frame.size = newSize;
    uslaunchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();
    uslaunchParams.frame.intensityBuffer = (float*)intensityBuffer.d_pointer();

    // and re-set the camera, since aspect may have changed
    collide_models_id.resize(uslaunchParams.maxBounce * sizeof(uint8_t));
    uslaunchParams.needle.collide_models_id = (uint8_t*)collide_models_id.d_pointer();

    collide_models_pos.resize(uslaunchParams.maxBounce * sizeof(vec3f));
    uslaunchParams.needle.collide_models_pos = (vec3f*)collide_models_pos.d_pointer();

}

void USRenderer::paraResize(int w, int h) {
    this->pixels.resize(w * h);
    this->collide_id.resize(uslaunchParams.maxBounce,-1);
    this->collide_pos.resize(uslaunchParams.maxBounce);
}


void USRenderer::postProcess() {
   postProcess_gpu(
       (uint8_t*)noiseBuffer.d_pointer(),
       noiseSize.x,
       noiseSize.y,
       GlobalConfig::SampleNum ,
       (float*)intensityBuffer.d_pointer(),
       (uint32_t*)colorBuffer.d_pointer(),
       (uint32_t*)postprocessBuffer.d_pointer(),
       uslaunchParams.frame.size.x, uslaunchParams.frame.size.y
       , 14, 3, uslaunchParams.needle.relaAngle, stream);
}

void USRenderer::downloadPixels()
{
    colorBuffer.download(this->pixels.data(),
        uslaunchParams.frame.size.x * uslaunchParams.frame.size.y);
}

uint32_t* USRenderer::pixelsData() {
    return this->pixels.data();
}

void USRenderer::downloadCollideInfo()
{
    collide_models_id.download(this->collide_id.data(), uslaunchParams.maxBounce);
    collide_models_pos.download(this->collide_pos.data(), uslaunchParams.maxBounce);
    getCollideId();
}

void USRenderer::getCollideId() {
    std::vector<float> insecDistance;
    insecDistance.resize(uslaunchParams.maxBounce);
    this->now_collide_model = -1;
    this->now_collide_ovam = -1;
    this->now_ovam_pos = vec3f(0.0, 0.0, 0.0);
    if (this->collide_id[0] == -1) return;
    for (int i = 0; i < uslaunchParams.maxBounce; i++) {
        if (this->collide_id[i] >= 0) {
            insecDistance[i] = sqrt(
                (this->collide_pos[i].x - uslaunchParams.transducer.position.x) * (this->collide_pos[i].x - uslaunchParams.transducer.position.x)
                + (this->collide_pos[i].y - uslaunchParams.transducer.position.y) * (this->collide_pos[i].y - uslaunchParams.transducer.position.y)
                + (this->collide_pos[i].z - uslaunchParams.transducer.position.z) * (this->collide_pos[i].z - uslaunchParams.transducer.position.z));
        }
    }

    //qDebug() << "nearest: " << insecDistance[0] << "  " << "needle: " << uslaunchParams.needle.relaDepth * uslaunchParams.frame.size.x / 2 ;

    for (int i = 0; i < uslaunchParams.maxBounce - 1; i++) {
        if (insecDistance[i] > 0  && insecDistance[i+1] > 0) {
            if (uslaunchParams.needle.relaDepth * uslaunchParams.frame.size.x / 2.0 > insecDistance[i]
                && uslaunchParams.needle.relaDepth * uslaunchParams.frame.size.x / 2.0 < insecDistance[i + 1]) {
                this->now_collide_model = this->collide_id[i];
                if (this->collide_id[i] == this->collide_id[i + 1] && int(this->collide_id[i]) >= 6) {
                    this->now_collide_ovam = this->collide_id[i];
                    this->now_ovam_pos = this->collide_pos[i];
                    return;
                }
            }
        }
    }
}

int USRenderer::getCollideModel() {
    return this->now_collide_model;
}

void USRenderer::clear() {
    this->resize(vec2i(this->uslaunchParams.frame.size.x, this->uslaunchParams.frame.size.y));
}

void USRenderer::run() {

}

float USRenderer::getNeedleEndX(float x) {
    return x + uslaunchParams.needle.relaDepth * sin((uslaunchParams.needle.relaAngle / 180.0) * PI);
}
float USRenderer::getNeedleEndY(float y) {
    return y + uslaunchParams.needle.relaDepth * cos((uslaunchParams.needle.relaAngle / 180.0) * PI);
}
//
//void processInput(GLFWwindow* window, USRenderer* render) {
//    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
//        glfwSetWindowShouldClose(window, true);
//    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
//        render->changeTransducer(1.0f, { 1.0f, 0.0f, 0.0f });
//    }
//    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
//        render->changeTransducer(-1.0f, { 1.0f, 0.0f, 0.0f });
//    }
//    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
//        render->changeTransducer(-1.0f, { 0.0f, 1.0f, 0.0f });
//    }
//    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
//        render->changeTransducer(1.0f, { 0.0f, 1.0f, 0.0f });
//    }
//    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
//        render->changeTransducer(1.0f, { 0.0f, 0.0f, 1.0f });
//    }
//    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
//        render->changeTransducer(-1.0f, { 0.0f, 0.0f, 1.0f });
//    }
//    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
//    {
//        render->changeNeedle(0.002f);
//    }
//    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
//    {
//        render->changeNeedle(-0.002f);
//    }
//}
//
//
//void USRenderer::run() {
//    //int width = this->uslaunchParams.frame.size.x;
//    //int height = this->uslaunchParams.frame.size.y;
//    //this->resize(vec2i(width, height));
//
//    //assert(glfwInit() && "GLFW initialization failed");
//    //GLFWwindow* window = glfwCreateWindow(1500, 900, "OvaSim", NULL, NULL);
//    //if (!window) {
//    //    glfwTerminate();
//    //    throw std::runtime_error("window has not be created");
//    //}
//    //glfwMakeContextCurrent(window);
//    //this->pixels.resize(width * height);
//    //this->collide_id.resize(uslaunchParams.maxBounce);
//    //this->collide_pos.resize(uslaunchParams.maxBounce);
//    //gladLoadGL();
//    //GLuint texture;
//    //glGenTextures(1, &texture);
//    //glBindTexture(GL_TEXTURE_2D, texture);
//    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//    //int frame = 0;
//    ////glfwSwapInterval(1);
//    //double lastTime = glfwGetTime();
//
//    //while (!glfwWindowShouldClose(window)) {
//    //    glfwSetKeyCallback(window, key_callback);
//    //    frame++;
//    //    double currentTime = glfwGetTime();
//    //    if ((currentTime - lastTime) >= 1.0) {
//    //        qDebug() << "fps:" << frame << "time:" << currentTime << "absorb:" << needleSwicth ;
//    //        //for (int _test = 0; _test < uslaunchParams.maxBounce; _test++) {
//    //        //    qDebug() << " " << int(this->collide_id[_test]);
//    //        //}
//    //        frame = 0;
//    //        lastTime = currentTime;
//    //    }
//    //    //uint32_t rgba = 0xff000000
//    //    //    | (120 << 0) | (int(frame % 256) << 8) | (200 << 16);
//    //    //std::fill(this->pixels.begin(), this->pixels.end(), rgba);
//    //    //this->clear();
//    //    if (needleSwicth)this->updateAccel();
//    //    this->resize(vec2i(width, height));
//    //    this->render();
//    //    this->postProcess();
//    //    this->downloadPixels(this->pixels.data());
//    //    this->downloadCollideInfo(this->collide_id.data(), this->collide_pos.data());
//    //    this->now_collide_ovam = getCollideOvamId();
//    //    float needelStart_x = 0.0;
//    //    float needelStart_y = -1.0;
//    //    float needleEnd_x = needelStart_x + uslaunchParams.needle.relaDepth * sin((uslaunchParams.needle.relaAngle / 180.0) * PI);
//    //    float needleEnd_y = needelStart_y + uslaunchParams.needle.relaDepth * cos((uslaunchParams.needle.relaAngle / 180.0) * PI);
//
//    //    //������ͼ
//    //    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, this->pixels.data());
//    //    glClear(GL_COLOR_BUFFER_BIT);
//    //    glEnable(GL_TEXTURE_2D);
//    //    glBindTexture(GL_TEXTURE_2D, texture);
//    //    glBegin(GL_QUADS);
//    //    glTexCoord2f(0, 0); glVertex2f(-1, -1);
//    //    glTexCoord2f(1, 0); glVertex2f(1, -1);
//    //    glTexCoord2f(1, 1); glVertex2f(1, 1);
//    //    glTexCoord2f(0, 1); glVertex2f(-1, 1);
//    //    glEnd();
//    //    glDisable(GL_TEXTURE_2D);
//    //    //�������
//    //    glBegin(GL_LINES);
//    //    glColor3f(0.6, 0.6, 0.6);
//    //    glVertex2f(needelStart_x, needelStart_y);
//    //    glVertex2f(needleEnd_x, needleEnd_y);
//    //    glEnd();
//
//    //    processInput(window, this);
//    //    glfwSwapBuffers(window);
//    //    glfwPollEvents();
//    //    //break;
//    //}
//    //glfwTerminate();
//}
