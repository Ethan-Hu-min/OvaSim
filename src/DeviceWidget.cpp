#include "DeviceWidget.h"
#include <QLabel>


DeviceWidget::DeviceWidget(QWidget *parent)
    : QWidget(parent)
{
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
    hdStartScheduler();
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        //hduPrintError(stderr, &error, "Failed to start the scheduler");
    }
    
    ui.device_para_name->setText(hdGetString(HD_DEVICE_MODEL_TYPE));
    // Schedule the frictionless plane callback, which will then run at 
    // servoloop rates and command forces if the user penetrates the plane.
    InfoCallback = hdScheduleAsynchronous(
        GetDeviceInfoCallback, &DeviceWidgetInfo, HD_DEFAULT_SCHEDULER_PRIORITY);

    connect(timer, SIGNAL(timeout()), this, SLOT(showData()));
    timer->start(33);
}

DeviceWidget::~DeviceWidget()

{
    delete timer;
    hdStopScheduler();
    hdUnschedule(InfoCallback);
    hdDisableDevice(hHD);
}

void DeviceWidget::closeEvent(QCloseEvent*) {
    emit ExitWin();
}

void DeviceWidget::showData() {

        if (!hdWaitForCompletion(InfoCallback, HD_WAIT_CHECK_STATUS))
    {
        fprintf(stderr, "\nThe main scheduler callback has exited\n");
        qDebug() << "The main scheduler callback has exited\n";
    }
    
        ui.device_para_x_set->setText(QString::number(DeviceWidgetInfo.position[0], 'f', 2));
        ui.device_para_y_set->setText(QString::number(DeviceWidgetInfo.position[1], 'f', 2));
        ui.device_para_z_set->setText(QString::number(DeviceWidgetInfo.position[2], 'f', 2));
        ui.device_para_gamma_set->setText(QString::number(DeviceWidgetInfo.angles[0], 'f', 2));
        ui.device_para_alpha_set->setText(QString::number(DeviceWidgetInfo.angles[2], 'f', 2));
        ui.device_para_beta_set->setText(QString::number(DeviceWidgetInfo.angles[1], 'f', 2));


}
