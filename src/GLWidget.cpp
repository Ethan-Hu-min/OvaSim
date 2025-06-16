#include "GLWidget.h"    
#include <GlobalConfig.h>
#include <QKeyEvent>
#include <QTimer>
#include<qregularexpression.h>
GLWidget::GLWidget(QWidget* parent)
	: QOpenGLWidget(parent)
{
    setAttribute(Qt::WA_DeleteOnClose);
    //hHD = hdInitDevice(HD_DEFAULT_DEVICE);
    //if (HD_DEVICE_ERROR(error = hdGetError()))
    //{
    //    qDebug() << " Failed to initialize haptic device ";
    //    //hduPrintError(stderr, &error, "Failed to initialize haptic device");
    //}
    //// Start the servo scheduler and enable forces.
    //hdEnable(HD_FORCE_OUTPUT);
    //hdStartScheduler();
    //if (HD_DEVICE_ERROR(error = hdGetError()))
    //{
    //    qDebug() << "Failed to start the scheduler";
    //    //hduPrintError(stderr, &error, "Failed to start the scheduler");
    //}
    serial = new QSerialPort(this);
    serialTimer = new QTimer();
    connect(serialTimer, SIGNAL(timeout()), this, SLOT(sendCommand()));
    openSerialPort();
    //originTransducerPos = vec3f(50.0f, -200.0f, 100.0f);
    originTransducerPos = vec3f(38.0f, -221.0f, 130.0f);

    //vec3f endPos = vec3f()

    originTransducerDir = vec3f(-0.586f, 0.785991f, 0.197033f);
    originTransducerHor = vec3f(0.602517f, 0.260058f, 0.754548f);
    originTransducerVer = normalize(cross(originTransducerHor, originTransducerDir));

    originTransducerAngle = 120.0;
    createRenderer();
}

GLWidget::~GLWidget() {
	//if(imageData != nullptr)delete[] imageData;
 //   if (texture != nullptr) delete texture;
	//imageData = nullptr;
    qDebug() << "!!!delete GLwidget\n";
    if (usRenderer != nullptr) {
        delete usRenderer;
        usRenderer = nullptr;
    }
    if (serial != nullptr) {
        delete serial;
        serial = nullptr;
    }
    if (serialTimer != nullptr) {
        delete serialTimer;
        serialTimer = nullptr;
    }


    //hdStopScheduler();
    //hdUnschedule(hForceGLCallback);
    //hdDisableDevice(hHD);
   
}

void GLWidget::closeEvent(QCloseEvent*) {
    //hdStopScheduler();
    //hdUnschedule(hForceCallback);
    //hdDisableDevice(hHD);
    //if (startRender) {
    //    hdStopScheduler();
    //}
    //hdUnschedule(hForceGLCallback);
    //hdDisableDevice(hHD);
    emit ExitWin();
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
    usRenderer->setNeedle(GlobalConfig::needleAngle, GlobalConfig::needleDepth);
    usRenderer->setTransducer(scene->transducer);
    usRenderer->initTexture();
    qDebug() << "[INFO] Init textures success";
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
                usRenderer->changeNeedle(0.001f * fps);
                break;
            case Qt::Key_2:
                usRenderer->changeNeedle(-0.001f * fps);
                break;
            case Qt::Key_X:
                if (!event->isAutoRepeat()) {
                    needleSwitch = true;
                    emit needleSwitchSignal(QString("OPEN"));
                }
            break;
            }
            break;
        case 2:
            switch (event->key())
            {
            case Qt::Key_X:
                if (!event->isAutoRepeat()) {
                    needleSwitch = true;
                    emit needleSwitchSignal(QString("OPEN"));
                }
            break;
            }
         break;
        }
    }

}

void GLWidget::keyReleaseEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_X) {
        // 同样检查自动重复
        if (!event->isAutoRepeat()) {
            needleSwitch = false;
            emit needleSwitchSignal(QString("CLOSE"));
        }
    }
}

void GLWidget::openSerialPort() {
    serial->setPortName("COM3");
    serial->setBaudRate(QSerialPort::Baud57600);
    serial->setDataBits(QSerialPort::Data8);
    serial->setParity(QSerialPort::NoParity);
    serial->setStopBits(QSerialPort::OneStop);

    if (serial->open(QIODevice::ReadWrite)) {
        connect(serial, SIGNAL(readyRead()), this, SLOT(onReadyRead()));
        serialTimer->start(30); // 30ms 定时发送
        qDebug() << "串口已打开";
    }
    else {
        qDebug() << "打开串口失败:" << serial->errorString();
    }
}

void GLWidget::sendCommand() {
    serial->write("$?\r\n"); // 发送指令
}

void GLWidget::onReadyRead() {
    serialBuffer += serial->readAll(); // 读取数据到缓冲区
    //qDebug() << "读取数据";
    // 按换行符分割数据
    while (serialBuffer.contains("\r\n")) {
        int endIndex = serialBuffer.indexOf("\r\n");
        QByteArray line = serialBuffer.left(endIndex);
        serialBuffer = serialBuffer.mid(endIndex + 2); // 移除已处理数据
        processLine(line);
    }
}

void GLWidget::processLine(const QByteArray& line) {
    QString data = QString::fromLatin1(line);
    //qDebug() << "处理数据" << data;
    QRegularExpression regex(R"((?:^|[, ])id=(\d+),(\d+)(?:[, ]|$))");
    QRegularExpressionMatch match = regex.match(data);

    if (match.hasMatch()) {
        bool ok;
        int idValue = match.captured(2).toInt(&ok); // 提取第二个捕获组
        if (ok) {
            needleDepth = idValue;
        }
        else {
            qDebug() << "Invalid integer format";
        }
    }
    else {
        qDebug() << "Pattern not found: " << data;
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

        
        if (needleSwitch && collideModel > 6) {
            usRenderer->updateAccel();
            tubeCapacity = (tubeCapacity + 1) % 100;
            emit nowCapacity(tubeCapacity);
        }
        
        
        

        usRenderer->resize(vec2i(frameSizeWidth, frameSizeHeight));
        usRenderer->render();
        usRenderer->postProcess();
        usRenderer->downloadPixels();
        usRenderer->downloadCollideInfo();
        collideModel = usRenderer->getCollideModel();
        //qDebug() << "[INFO]Model: " << collideModel;
        if(collideModel < 5 && collideModel >=0)emit collideInfo(collideModel);//0,1,2,3,4 for other model;
        if (collideModel < 7 && needleSwitch)emit collideInfo(10);//error using needleswitch;
        

        float needleStartX = 0.0;
        float needleStartY = -1.0;
        float needleEndX = usRenderer->getNeedleEndX(needleStartX);
        float needleEndY = usRenderer->getNeedleEndY(needleStartY);

        float scaleFactor = 1.5f;//画面缩放因子


        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frameSizeWidth, frameSizeHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, usRenderer->pixelsData());
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, displayTexture);

        // 保存当前的变换矩阵
        glPushMatrix();
        // 平移到缩放中心
        glTranslatef(0.0f, -1.0f, 0.0f);
        // 应用缩放变换
        glScalef(scaleFactor, scaleFactor, 1.0f);
        // 平移回原来的位置
        glTranslatef(0.0f, 1.0f, 0.0f);

        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f(1, -1);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
        glTexCoord2f(0, 1); glVertex2f(-1, 1);
        glEnd();

        // 恢复原来的变换矩阵
        glPopMatrix();
        glDisable(GL_TEXTURE_2D);

        glBegin(GL_LINES);
        glColor3f(0.9, 0.9, 0.9);
        glVertex2f(needleStartX, needleStartY);
        glVertex2f(needleEndX + 0.01, needleEndY - 0.01);
        glEnd();

        glBegin(GL_LINES);
        glColor3f(0.9, 0.9, 0.9);
        glVertex2f(needleStartX, needleStartY+0.02);
        glVertex2f(needleEndX + 0.01, needleEndY + 0.01);
        glEnd();

        glPointSize(3.0);                   // 点大小
        glColor3f(1.0, 1.0, 1.0);           // 黄色
        glBegin(GL_POINTS);
        glVertex2f(needleEndX  , needleEndY);               // 原线段末端延伸点
        glVertex2f(needleEndX  , needleEndY + 0.02); // 平移线段末端延伸点
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
            float changeYaw = DeviceWidgetInfo.angles[0] - originDeviceAngles.x;
            float changePitch = DeviceWidgetInfo.angles[1] - originDeviceAngles.y;
            float changeRow = DeviceWidgetInfo.angles[2] - originDeviceAngles.z;
            usRenderer->changeTransducerAbs(floor(changeYaw), -floor(changePitch), floor(changeRow), originTransducerDir, originTransducerHor, originTransducerVer);
            //usRenderer->changeTransducerAbs(changeRow, -changePitch, 0.0);
            usRenderer->changeNeedleAbs((1024 - needleDepth) / 1024.0 );
        }

        if (!hdWaitForCompletion(hForceGLCallback, HD_WAIT_CHECK_STATUS))
        {
            qDebug() << "The main scheduler callback has exited";
        }

        update();
    }
}



void GLWidget::setStartRenderTrue() {

    DeviceWidgetInfo.originPos = hduVector3Dd(0, -60, -60);
    DeviceWidgetInfo.originAngle = hduVector3Dd(0, 0, 0);
    DeviceWidgetInfo.dampingForce = 0.3;
    DeviceWidgetInfo.dampingTorque = 1.0;

    hForceGLCallback = hdScheduleAsynchronous(
        ForceGLCallback, &DeviceWidgetInfo, HD_DEFAULT_SCHEDULER_PRIORITY);


    qDebug() << "device has used";
    startRender = true;
    QTimer::singleShot(100, this, &GLWidget::setOriginDevice);

    update();
}

void GLWidget::setStartRenderFalse() {
    startRender = false;

    update();
}

void GLWidget::setOriginDevice() {
    originDevicePos = vec3f(DeviceWidgetInfo.position[0], DeviceWidgetInfo.position[1], DeviceWidgetInfo.position[2]);
    originDeviceAngles = vec3f(DeviceWidgetInfo.angles[0], DeviceWidgetInfo.angles[1], DeviceWidgetInfo.angles[2]);
    DeviceWidgetInfo.originPos = hduVector3Dd(originDevicePos.x, originDevicePos.y, originDevicePos.z);
    DeviceWidgetInfo.originAngle = hduVector3Dd(originDeviceAngles.x, originDeviceAngles.y, originDeviceAngles.z);
}

