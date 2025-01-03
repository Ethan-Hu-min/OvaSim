#include "GLWidget.h"    
#include <GlobalConfig.h>

GLWidget::GLWidget(QWidget* parent)
	: QOpenGLWidget(parent)
{
    createRenderer();
}

GLWidget::~GLWidget() {
	//if(imageData != nullptr)delete[] imageData;
 //   if (texture != nullptr) delete texture;
	//imageData = nullptr;
    qDebug() << "imageData clean\n";
    if(usRenderer != nullptr) delete usRenderer;
   
}

void GLWidget::createRenderer() {
    Scene* scene = new Scene();
    scene->setExampleName("example1");
    scene->parseConfig("example1.scene");
    scene->loadModels();
    scene->createWorldModels();
    qDebug() << "models nums: " << scene->worldmodel.size();
    scene->setTransducer(vec3f(50.0f, -200.0f, 100.0f),
        vec3f(0.197562f, 0.729760f, -0.654538f), vec3f(0.728110f, -0.556300f, -0.400482f), GlobalConfig::transducerNums, 120.0, 512.0, 512.0);
    usRenderer = new USRenderer(scene);
    usRenderer->setNeedle(-30.0, 0.0);
    usRenderer->setTransducer(scene->transducer);
    frameSizeHeight = GlobalConfig::transducerNums;
    frameSizeWidth = GlobalConfig::transducerNums;
}

//void GLWidget::initTextures()
//{
//
//    imageData = new uchar[this->height() * this->width() * 4];
//    for (int i = 0; i < this->height() * this->width() * 4; i++) {
//        imageData[i] = static_cast<uchar>(50);
//    }
//
//	texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
//
//	//重复使用纹理坐标
//	//纹理坐标(1.1, 1.2)与(0.1, 0.2)相同
//	texture->setWrapMode(QOpenGLTexture::Repeat);
//
//    texture->setFormat(QOpenGLTexture::RGBA8U);
//	//设置纹理大小
//	texture->setSize(this->width(), this->height());
//	//分配储存空间
//	texture->allocateStorage();
//  
//}

//void GLWidget::initShaders()
//{
//    //纹理坐标
//    texCoords.append(QVector2D(0, 1)); //左上
//    texCoords.append(QVector2D(1, 1)); //右上
//    texCoords.append(QVector2D(0, 0)); //左下
//    texCoords.append(QVector2D(1, 0)); //右下
//    //顶点坐标
//    vertices.append(QVector3D(-1, -1, 1));//左下
//    vertices.append(QVector3D(1, -1, 1)); //右下
//    vertices.append(QVector3D(-1, 1, 1)); //左上
//    vertices.append(QVector3D(1, 1, 1));  //右上
//    QOpenGLShader* vshader = new QOpenGLShader(QOpenGLShader::Vertex, this);
//    const char* vsrc =
//        "attribute vec4 vertex;\n"
//        "attribute vec2 texCoord;\n"
//        "varying vec2 texc;\n"
//        "void main(void)\n"
//        "{\n"
//        "    gl_Position = vertex;\n"
//        "    texc = texCoord;\n"
//        "}\n";
//    vshader->compileSourceCode(vsrc);//编译顶点着色器代码
//
//    QOpenGLShader* fshader = new QOpenGLShader(QOpenGLShader::Fragment, this);
//    const char* fsrc =
//        "uniform sampler2D texture;\n"
//        "varying vec2 texc;\n"
//        "void main(void)\n"
//        "{\n"
//        "    gl_FragColor = texture2D(texture,texc);\n"
//        "}\n";
//    fshader->compileSourceCode(fsrc); //编译纹理着色器代码
//
//    program.addShader(vshader);//添加顶点着色器
//    program.addShader(fshader);//添加纹理碎片着色器
//    program.bindAttributeLocation("vertex", 0);//绑定顶点属性位置
//    program.bindAttributeLocation("texCoord", 1);//绑定纹理属性位置
//    // 链接着色器管道
//    if (!program.link())
//        close();
//    // 绑定着色器管道
//    if (!program.bind())
//        close();
//}


void GLWidget::initializeGL() {
   
    initializeOpenGLFunctions(); //初始化OPenGL功能函数
    glClearColor(174 / 255.0, 208 / 255.0, 238 / 255.0, 1.0);    //设置背景
    glEnable(GL_TEXTURE_2D);     //设置纹理2D功能可用
    //initTextures();              //初始化纹理设置
    //initShaders();               //初始化shaders
    
    usRenderer->resize(vec2i(frameSizeWidth, frameSizeHeight));
    usRenderer->paraResize(frameSizeWidth, frameSizeHeight);


    glGenTextures(1, &displayTexture);
    glBindTexture(GL_TEXTURE_2D, displayTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    QTime currentTime = QTime::currentTime();
    lastTime = currentTime.msecsSinceStartOfDay();
}


void GLWidget::paintGL()
{

    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //清除屏幕缓存和深度缓冲
    //program.enableAttributeArray(0);
    //program.enableAttributeArray(1);
    //program.setAttributeArray(0, vertices.constData());
    //program.setAttributeArray(1, texCoords.constData());
    //program.setUniformValue("texture", 0); //将当前上下文中位置的统一变量设置为value
    //texture->bind();  //绑定纹理
    //glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);//绘制纹理
    //texture->release(); //释放绑定的纹理
    //texture->destroy(); //消耗底层的纹理对象
    //texture->create();
    frameNo++;
    QTime currentTime = QTime::currentTime();
    int nowTime = currentTime.msecsSinceStartOfDay();
    if ((nowTime - lastTime) >= 1000) {
        fps = frameNo;
        frameNo = 0;
        qDebug() << "fps: " << fps;
        lastTime = nowTime;

    }
    usRenderer->resize(vec2i(frameSizeWidth, frameSizeHeight));
    usRenderer->render();
    usRenderer->postProcess();
    usRenderer->downloadPixels();


    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frameSizeWidth, frameSizeHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, usRenderer->pixelsData());
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, displayTexture);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(-1, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);

}


void GLWidget::resizeGL(int w, int h) {
    this->glViewport(0, 0, w, h);
}


void GLWidget::setImage()
{
    //texture->setData(QImage(imageData,  this->height(), this->width(), QImage::Format_RGB32)); //设置纹理图像
    update();
}




//int main(int argc, char *argv[])
//{
//    try
//    {
//        //
//        // Initialize CUDA and create OptiX context
//        //
//        OptixDeviceContext context = nullptr;
//        {
//            // Initialize CUDA
//            CUDA_CHECK(cudaFree(0));
//
//            CUcontext cuCtx = 0;  // zero means take the current context
//            OPTIX_CHECK(optixInit());
//            OptixDeviceContextOptions options = {};
//            options.logCallbackFunction = &context_log_cb;
//            options.logCallbackLevel = 4;
//            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
//        }
//
//        //
//        // Create module
//        //
//        OptixModule module = nullptr;
//        OptixPipelineCompileOptions pipeline_compile_options = {};
//        {
//            OptixModuleCompileOptions module_compile_options = {};
//#if !defined(NDEBUG)
//            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
//            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
//#endif
//            pipeline_compile_options.usesMotionBlur = false;
//            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
//            pipeline_compile_options.numPayloadValues = 2;
//            pipeline_compile_options.numAttributeValues = 2;
//            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
//            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
//
//            size_t      inputSize = 0;
//            const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "draw_solid_color.cu", inputSize);
//
//            OPTIX_CHECK_LOG(optixModuleCreate(
//                context,
//                &module_compile_options,
//                &pipeline_compile_options,
//                input,
//                inputSize,
//                LOG, &LOG_SIZE,
//                &module
//            ));
//        }
//
//        //
//        // Create program groups, including NULL miss and hitgroups
//        //
//        OptixProgramGroup raygen_prog_group = nullptr;
//        OptixProgramGroup miss_prog_group = nullptr;
//        {
//            OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
//
//            OptixProgramGroupDesc raygen_prog_group_desc = {}; //
//            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
//            raygen_prog_group_desc.raygen.module = module;
//            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__draw_solid_color";
//            OPTIX_CHECK_LOG(optixProgramGroupCreate(
//                context,
//                &raygen_prog_group_desc,
//                1,   // num program groups
//                &program_group_options,
//                LOG, &LOG_SIZE,
//                &raygen_prog_group
//            ));
//
//            // Leave miss group's module and entryfunc name null
//            OptixProgramGroupDesc miss_prog_group_desc = {};
//            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
//            OPTIX_CHECK_LOG(optixProgramGroupCreate(
//                context,
//                &miss_prog_group_desc,
//                1,   // num program groups
//                &program_group_options,
//                LOG, &LOG_SIZE,
//                &miss_prog_group
//            ));
//        }
//
//        //
//        // Link pipeline
//        //
//        OptixPipeline pipeline = nullptr;
//        {
//            const uint32_t    max_trace_depth = 0;
//            OptixProgramGroup program_groups[] = { raygen_prog_group };
//
//            OptixPipelineLinkOptions pipeline_link_options = {};
//            pipeline_link_options.maxTraceDepth = max_trace_depth;
//            OPTIX_CHECK_LOG(optixPipelineCreate(
//                context,
//                &pipeline_compile_options,
//                &pipeline_link_options,
//                program_groups,
//                sizeof(program_groups) / sizeof(program_groups[0]),
//                LOG, &LOG_SIZE,
//                &pipeline
//            ));
//
//            OptixStackSizes stack_sizes = {};
//            for (auto& prog_group : program_groups)
//            {
//                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
//            }
//
//            uint32_t direct_callable_stack_size_from_traversal;
//            uint32_t direct_callable_stack_size_from_state;
//            uint32_t continuation_stack_size;
//            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
//                0,  // maxCCDepth
//                0,  // maxDCDEpth
//                &direct_callable_stack_size_from_traversal,
//                &direct_callable_stack_size_from_state, &continuation_stack_size));
//            OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
//                direct_callable_stack_size_from_state, continuation_stack_size,
//                2  // maxTraversableDepth
//            ));
//        }
//
//        //
//        // Set up shader binding table
//        //
//        OptixShaderBindingTable sbt = {};
//        {
//            CUdeviceptr  raygen_record;
//            const size_t raygen_record_size = sizeof(RayGenSbtRecord);
//            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
//            RayGenSbtRecord rg_sbt;
//            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
//            rg_sbt.data = { 0.462f, 0.725f, 0.f };
//            CUDA_CHECK(cudaMemcpy(
//                reinterpret_cast<void*>(raygen_record),
//                &rg_sbt,
//                raygen_record_size,
//                cudaMemcpyHostToDevice
//            ));
//
//            CUdeviceptr miss_record;
//            size_t      miss_record_size = sizeof(MissSbtRecord);
//            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
//            MissSbtRecord ms_sbt;
//            OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
//            CUDA_CHECK(cudaMemcpy(
//                reinterpret_cast<void*>(miss_record),
//                &ms_sbt,
//                miss_record_size,
//                cudaMemcpyHostToDevice
//            ));
//
//            sbt.raygenRecord = raygen_record;
//            sbt.missRecordBase = miss_record;
//            sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
//            sbt.missRecordCount = 1;
//        }
//
//        sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);
//
//        //
//        // launch
//        //
//        {
//            CUstream stream;
//            CUDA_CHECK(cudaStreamCreate(&stream));
//
//            Params params;
//            params.image = output_buffer.map();
//            params.image_width = width;
//
//            CUdeviceptr d_param;
//            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
//            CUDA_CHECK(cudaMemcpy(
//                reinterpret_cast<void*>(d_param),
//                &params, sizeof(params),
//                cudaMemcpyHostToDevice
//            ));
//
//            OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));
//            CUDA_SYNC_CHECK();
//
//            output_buffer.unmap();
//            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
//        }
//
//        //
//        // Display results
//        //
//        {
//            sutil::ImageBuffer buffer;
//            buffer.data = output_buffer.getHostPointer();
//            buffer.width = width;
//            buffer.height = height;
//            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
//            if (outfile.empty())
//                sutil::displayBufferWindow(argv[0], buffer);
//            else
//                sutil::saveImage(outfile.c_str(), buffer, false);
//        }
//
//        //
//        // Cleanup
//        //
//        {
//            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
//            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
//
//            OPTIX_CHECK(optixPipelineDestroy(pipeline));
//            OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
//            OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
//            OPTIX_CHECK(optixModuleDestroy(module));
//
//            OPTIX_CHECK(optixDeviceContextDestroy(context));
//        }
//    }
//    catch (std::exception& e)
//    {
//        qDebug() << "Caught exception: " << e.what() << "\n";
//        return 1;
//    }
//    return 0;
//
//}