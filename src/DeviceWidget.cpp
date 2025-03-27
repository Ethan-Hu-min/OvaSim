#include "DeviceWidget.h"
#include <QLabel>
#include <QDateTime>
#include <GlobalConfig.h>
#include <QKeyEvent>

DeviceWidget::DeviceWidget(QWidget *parent)
    : QWidget(parent)
{

    QDateTime currentDateTime = QDateTime::currentDateTime();
    QString formattedDateTime = currentDateTime.toString("yyyy-MM-dd-HH-mm");
    QString qfilename = formattedDateTime + ".log"; 
    qDebug() << "nowlog:" << qfilename;
    std::string logFileName = GlobalConfig::logPath +std::string(qfilename.toLocal8Bit());

    deviceLog = spdlog::basic_logger_mt("deviceLog", logFileName);

    setAttribute(Qt::WA_DeleteOnClose);
    ui.setupUi(this);
    this->timer = new QTimer;

    // Initialize the default haptic device.
    hHD = hdInitDevice(HD_DEFAULT_DEVICE);
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        //hduPrintError(stderr, &error, "Failed to initialize haptic device");
    }
    // Start the servo scheduler and enable forces.
    hdEnable(HD_FORCE_OUTPUT);

    
    ui.device_para_name->setText(hdGetString(HD_DEVICE_MODEL_TYPE));

    DeviceWidgetInfo.originPos = hduVector3Dd(0,0,0);
    DeviceWidgetInfo.originAngle = hduVector3Dd(0, 0, 0); 

    DeviceWidgetInfo.dampingForce = 0.2;
    DeviceWidgetInfo.dampingTorque = 1.0;



    ui.device_origin_damping->setSingleStep(0.1);
    ui.device_origin_damping_torque->setSingleStep(0.1);
    ui.device_origin_damping_torque->setValue(DeviceWidgetInfo.dampingTorque);
    ui.device_origin_damping->setValue(DeviceWidgetInfo.dampingForce);
    ui.device_origin_damping_torque->setValue(DeviceWidgetInfo.dampingTorque);

    ui.device_origin_x->setRange(-360, 360);
    ui.device_origin_x->setSingleStep(1.0);
    ui.device_origin_x->setValue(DeviceWidgetInfo.originPos[0]);

    ui.device_origin_y->setRange(-360, 360);
    ui.device_origin_y->setSingleStep(1.0);
    ui.device_origin_y->setValue(DeviceWidgetInfo.originPos[1]);

    ui.device_origin_z->setRange(-360, 360);
    ui.device_origin_z->setSingleStep(1.0);
    ui.device_origin_z->setValue(DeviceWidgetInfo.originPos[2]);

    ui.device_origin_gamma->setRange(-360, 360);
    ui.device_origin_gamma->setSingleStep(1.0);
    ui.device_origin_gamma->setValue(DeviceWidgetInfo.originAngle[0]);

    ui.device_origin_beta->setRange(-360, 360);
    ui.device_origin_beta->setSingleStep(1.0);
    ui.device_origin_beta->setValue(DeviceWidgetInfo.originAngle[1]);

    ui.device_origin_alpha->setRange(-360, 360);
    ui.device_origin_alpha->setSingleStep(1.0);
    ui.device_origin_alpha->setValue(DeviceWidgetInfo.originAngle[2]);

    serial = new QSerialPort(this);
    serialTimer = new QTimer();
    openSerialPort();
    connect(ui.device_record_button, SIGNAL(clicked(bool)), this, SLOT(on_clicked_record_button()));
    connect(ui.device_force_button, SIGNAL(clicked(bool)), this, SLOT(on_clicked_force_button()));

    connect(timer, SIGNAL(timeout()), this, SLOT(showData()));
    connect(serialTimer, SIGNAL(timeout()), this, SLOT(sendCommand()));
    timer->start(33);
    
}

DeviceWidget::~DeviceWidget()

{
    if(timer != nullptr)delete timer;
    if (serialTimer != nullptr)delete serialTimer;
}

void DeviceWidget::setOriginDevice() {
    originDevicePos = DeviceWidgetInfo.position;
    originDeviceAngles = DeviceWidgetInfo.angles;
    qDebug() <<"originPos " << originDevicePos[0] << originDevicePos[1] << originDevicePos[2];
    DeviceWidgetInfo.originPos = originDevicePos;
    DeviceWidgetInfo.originAngle = originDeviceAngles;
}



void DeviceWidget::closeEvent(QCloseEvent*) {
    //hdStopScheduler();
    //hdUnschedule(hForceCallback);
    //hdDisableDevice(hHD);
    if (forceButton) {
        hdStopScheduler();
    }
    hdUnschedule(hForceCallback);
    hdDisableDevice(hHD);
    emit ExitWin();
}


void DeviceWidget::openSerialPort() {
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

void DeviceWidget::sendCommand() {
    serial->write("$?\r\n"); // 发送指令
}

void DeviceWidget::onReadyRead() {
    serialBuffer += serial->readAll(); // 读取数据到缓冲区
    //qDebug() << "读取数据" << serialBuffer;
    // 按换行符分割数据
    while (serialBuffer.contains("\r\n")) {
        int endIndex = serialBuffer.indexOf("\r\n");
        QByteArray line = serialBuffer.left(endIndex);
        serialBuffer = serialBuffer.mid(endIndex + 2); // 移除已处理数据
        processLine(line);
    }
}

void DeviceWidget::processLine(const QByteArray& line) {
    QString data = QString::fromLatin1(line);
    //qDebug() << "处理数据" << data;
    QRegularExpression regex(R"((?:^|[, ])id=(\d+),(\d+)(?:[, ]|$))");
    QRegularExpressionMatch match = regex.match(data);

    if (match.hasMatch()) {
        bool ok;
        int idValue = match.captured(2).toInt(&ok); // 提取第二个捕获组
        if (ok) {
            needleDepth = idValue;
            ui.device_needle->setNum(needleDepth);
        }
        else {
            qDebug() << "Invalid integer format";
        }
    }
    else {
        qDebug() << "Pattern not found";
    }
}

void DeviceWidget::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_X) {
        // 检查是否为自动重复事件（避免长按时重复触发）
        if (!event->isAutoRepeat()) {
            footswitch = true;
            ui.device_foot->setText("OPEN");
            qDebug() << "脚踏按下，footswitch = true";
        }
    }
}

void DeviceWidget::keyReleaseEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_X) {
        // 同样检查自动重复
        if (!event->isAutoRepeat()) {
            footswitch = false;
            ui.device_foot->setText("CLOSE");
            qDebug() << "脚踏释放，footswitch = false";
        }
    }
}

void DeviceWidget::on_clicked_force_button() {
    if (forceButton) {
        forceButton = false;
        ui.device_force_button->setText("启动力反馈");

        hdStopScheduler();

    }
    else {
        forceButton = true;
        ui.device_force_button->setText("关闭力反馈");

        hdStartScheduler();
        if (HD_DEVICE_ERROR(error = hdGetError()))
        {
            //hduPrintError(stderr, &error, "Failed to start the scheduler");
        }
        // Schedule the frictionless plane callback, which will then run at 
        // servoloop rates and command forces if the user penetrates the plane.
        hForceCallback = hdScheduleAsynchronous(
            ForceDeviceCallback, &DeviceWidgetInfo, HD_DEFAULT_SCHEDULER_PRIORITY);

       QTimer::singleShot(100, this, &DeviceWidget::setOriginDevice);



        ui.device_origin_x->setValue(DeviceWidgetInfo.originPos[0]);
        ui.device_origin_y->setValue(DeviceWidgetInfo.originPos[1]);
        ui.device_origin_z->setValue(DeviceWidgetInfo.originPos[2]);

        ui.device_origin_gamma->setValue(DeviceWidgetInfo.originAngle[0]);
        ui.device_origin_beta->setValue(DeviceWidgetInfo.originAngle[1]);
        ui.device_origin_alpha->setValue(DeviceWidgetInfo.originAngle[2]);
    }
}

void DeviceWidget::on_clicked_record_button() {
    if (recordButton) {
        recordButton = false;
        ui.device_record_button->setText("开始记录");

    }
    else {
        recordButton = true;
        ui.device_record_button->setText("停止记录");
    }
}

void DeviceWidget::showData() {

    if (forceButton) {
        if (!hdWaitForCompletion(hForceCallback, HD_WAIT_CHECK_STATUS))
        {
            fprintf(stderr, "\nThe main scheduler callback has exited\n");
            qDebug() << "The main scheduler callback has exited\n";
        }

        DeviceWidgetInfo.dampingForce = ui.device_origin_damping->value();
        DeviceWidgetInfo.dampingTorque = ui.device_origin_damping_torque->value();
        //DeviceWidgetInfo.originPos[0] = ui.device_origin_x->value();
        //DeviceWidgetInfo.originPos[1] = ui.device_origin_y->value();
        //DeviceWidgetInfo.originPos[2] = ui.device_origin_z->value();
        //DeviceWidgetInfo.originAngle[0] = ui.device_origin_gamma->value();
        //DeviceWidgetInfo.originAngle[1] = ui.device_origin_beta->value();
        //DeviceWidgetInfo.originAngle[2] = ui.device_origin_alpha->value();
        qDebug() << DeviceWidgetInfo.originPos[0] << DeviceWidgetInfo.originPos[1] << DeviceWidgetInfo.originPos[2];




        ui.device_para_x_set->setText(QString::number(DeviceWidgetInfo.position[0], 'f', 2));
        ui.device_para_y_set->setText(QString::number(DeviceWidgetInfo.position[1], 'f', 2));
        ui.device_para_z_set->setText(QString::number(DeviceWidgetInfo.position[2], 'f', 2));
        ui.device_para_gamma_set->setText(QString::number(DeviceWidgetInfo.angles[0], 'f', 2));
        ui.device_para_alpha_set->setText(QString::number(DeviceWidgetInfo.angles[2], 'f', 2));
        ui.device_para_beta_set->setText(QString::number(DeviceWidgetInfo.angles[1], 'f', 2));

        ui.device_para_x_force->setText(QString::number(DeviceWidgetInfo.position_force[0], 'f', 2));
        ui.device_para_y_force->setText(QString::number(DeviceWidgetInfo.position_force[1], 'f', 2));
        ui.device_para_z_force->setText(QString::number(DeviceWidgetInfo.position_force[2], 'f', 2));

        ui.device_para_x_torque->setText(QString::number(DeviceWidgetInfo.angles_torque[0], 'f', 2));
        ui.device_para_y_torque->setText(QString::number(DeviceWidgetInfo.angles_torque[1], 'f', 2));
        ui.device_para_z_torque->setText(QString::number(DeviceWidgetInfo.angles_torque[2], 'f', 2));

        if (recordButton) {

            deviceLog->info("P: {:.2f}, {:.2f}, {:.2f}", DeviceWidgetInfo.position[0], DeviceWidgetInfo.position[1], DeviceWidgetInfo.position[2]);
            deviceLog->info("A: {:.2f}, {:.2f}, {:.2f}", DeviceWidgetInfo.angles[0], DeviceWidgetInfo.angles[1], DeviceWidgetInfo.angles[2]);
            deviceLog->info("F: {:.2f}, {:.2f}, {:.2f}", DeviceWidgetInfo.position_force[0], DeviceWidgetInfo.position_force[1], DeviceWidgetInfo.position_force[2]);
            deviceLog->info("T: {:.2f}, {:.2f}, {:.2f}", DeviceWidgetInfo.angles_torque[0], DeviceWidgetInfo.angles_torque[1], DeviceWidgetInfo.angles_torque[2]);
        }
    }
}
