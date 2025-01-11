#include "GLWidget.h"    
#include <GlobalConfig.h>
#include <QKeyEvent>

GLWidget::GLWidget(QWidget* parent)
	: QOpenGLWidget(parent)
{
    hHD = hdInitDevice(HD_DEFAULT_DEVICE);
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        qDebug() << " Failed to initialize haptic device ";
        //hduPrintError(stderr, &error, "Failed to initialize haptic device");
    }
    // Start the servo scheduler and enable forces.
    controlMode = 2;
    hdEnable(HD_FORCE_OUTPUT);
    hdStartScheduler();
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        qDebug() << "Failed to start the scheduler";
        //hduPrintError(stderr, &error, "Failed to start the scheduler");
    }
    


    originTransducerPos = vec3f(50.0f, -200.0f, 100.0f);
    originTransducerDir = vec3f(0.197562f, 0.729760f, -0.654538f);
    originTransducerVer = vec3f(0.728110f, -0.556300f, -0.400482f);
    originTransducerAngle = 120.0;
    createRenderer();
}

GLWidget::~GLWidget() {
	//if(imageData != nullptr)delete[] imageData;
 //   if (texture != nullptr) delete texture;
	//imageData = nullptr;
    qDebug() << "imageData clean\n";
    if(usRenderer != nullptr) delete usRenderer;

    hdStopScheduler();
    hdUnschedule(hSphereCallback);
    hdDisableDevice(hHD);
   
}

void GLWidget::createRenderer() {
    Scene* scene = new Scene();
    scene->setExampleName("example1");
    scene->parseConfig("example1.scene");
    scene->loadModels();
    scene->createWorldModels();
    qDebug() << "models nums: " << scene->worldmodel.size();
    scene->setTransducer(originTransducerPos,
        originTransducerDir, originTransducerVer, GlobalConfig::transducerNums, originTransducerAngle, 512.0, 512.0);
    usRenderer = new USRenderer(scene);
    usRenderer->setNeedle(-30.0, 0.0);
    usRenderer->setTransducer(scene->transducer);
    frameSizeHeight = GlobalConfig::transducerNums;
    frameSizeWidth = GlobalConfig::transducerNums;
}

void GLWidget::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_C )
    {
        if (controlMode == 1) {
            controlMode = 2;
            qDebug() << "now control mode  is  device";
        }
        else {
            controlMode = 1;
            qDebug() << "now control mode  is  keyboard";
        }
    }

    if (startRender) {
        switch (controlMode) {
        case 1:
            switch (event->key())
            {
            case Qt::Key_W:
                usRenderer->changeTransducer(1.0f, { 1.0f, 0.0f, 0.0f });
                break;
            case Qt::Key_S:
                usRenderer->changeTransducer(-1.0f, { 1.0f, 0.0f, 0.0f });
                break;
            case Qt::Key_A:
                usRenderer->changeTransducer(-1.0f, { 0.0f, 1.0f, 0.0f });
                break;
            case Qt::Key_D:
                usRenderer->changeTransducer(1.0f, { 0.0f, 1.0f, 0.0f });
                break;
            case Qt::Key_Q:
                usRenderer->changeTransducer(1.0f, { 0.0f, 0.0f, 1.0f });
                break;
            case Qt::Key_E:
                usRenderer->changeTransducer(-1.0f, { 0.0f, 0.0f, 1.0f });
                break;
            case Qt::Key_1:
                usRenderer->changeNeedle(0.002f);
                break;
            case Qt::Key_2:
                usRenderer->changeNeedle(-0.002f);
                break;
            case Qt::Key_X:
            {
                needleSwitch = !needleSwitch;
                if (this->needleSwitch)emit needleSwitchSignal(QString("open"));
                else emit needleSwitchSignal(QString("close"));
            }
            break;
            }
            break;
        case 2:
            switch (event->key())
            {
            case Qt::Key_1:
                usRenderer->changeNeedle(0.002f);
                break;
            case Qt::Key_2:
                usRenderer->changeNeedle(-0.002f);
                break;
            case Qt::Key_X:
            {
                needleSwitch = !needleSwitch;
                if (this->needleSwitch)emit needleSwitchSignal(QString("open"));
                else emit needleSwitchSignal(QString("close"));
            }
            break;
            }
         break;
        }
    }

}

void GLWidget::initializeGL() {
   
    initializeOpenGLFunctions(); //初始化OPenGL功能函数
    glClearColor(174 / 255.0, 208 / 255.0, 238 / 255.0, 1.0);    //设置背景
    glEnable(GL_TEXTURE_2D);     //设置纹理2D功能可用
    
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
    if (startRender) {
        frameNo++;
        QTime currentTime = QTime::currentTime();
        int nowTime = currentTime.msecsSinceStartOfDay();
        if ((nowTime - lastTime) >= 1000) {
            fps = frameNo;
            frameNo = 0;
            emit fpsChanged(fps);
            lastTime = nowTime;
            int nownums = usRenderer->getNowObtainedNums();
            emit ovamNumsSignal(nownums);
        }


        if (needleSwitch)usRenderer->updateAccel();

        usRenderer->resize(vec2i(frameSizeWidth, frameSizeHeight));
        usRenderer->render();
        usRenderer->postProcess();
        usRenderer->downloadPixels();
        usRenderer->downloadCollideInfo();


        float needleStartX = 0.0;
        float needleStartY = -1.0;
        float needleEndX = usRenderer->getNeedleEndX(needleStartX);
        float needleEndY = usRenderer->getNeedleEndY(needleStartY);


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

        glBegin(GL_LINES);
        glColor3f(0.6, 0.6, 0.6);
        glVertex2f(needleStartX, needleStartY);
        glVertex2f(needleEndX, needleEndY);
        glEnd();

    }
    else {
        glClearColor(100 / 255.0, 100 / 255.0, 100 / 255.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
    }
}


void GLWidget::resizeGL(int w, int h) {
    this->glViewport(0, 0, w, h);
}


void GLWidget::setImage()
{
    if (startRender) {
        if (controlMode == 2) {
            float changeRow = DeviceWidgetInfo.angles[0] - originDeviceAngles.x;
            float changePitch = DeviceWidgetInfo.angles[1] - originDeviceAngles.y;
            float changeYaw = DeviceWidgetInfo.angles[2] - originDeviceAngles.z;
            usRenderer->changeTransducerAbs(changeRow/2.0, changePitch/2.0, changeYaw/2.0);
        }
        if (!hdWaitForCompletion(hSphereCallback, HD_WAIT_CHECK_STATUS))
        {
            qDebug() << "The main scheduler callback has exited";
        }

        update();
    }
}

void GLWidget::setStartRenderTrue() {
    hSphereCallback = hdScheduleAsynchronous(
        PosSphereCallback, &DeviceWidgetInfo, HD_DEFAULT_SCHEDULER_PRIORITY);
    qDebug() << "device has used";
    startRender = true;
    setOriginDevice();
    update();
}

void GLWidget::setStartRenderFalse() {
    startRender = false;
    update();
}

void GLWidget::setOriginDevice() {
    originDevicePos = vec3f(DeviceWidgetInfo.position[0], DeviceWidgetInfo.position[1], DeviceWidgetInfo.position[2]);
    originDeviceAngles = vec3f(DeviceWidgetInfo.angles[0], DeviceWidgetInfo.angles[1], DeviceWidgetInfo.angles[2]);
}

