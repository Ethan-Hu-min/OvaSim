#ifndef DEVICEWIDGET_H
#define DEVICEWIDGET_H

#include "UseDevice.h"

#include "ui_DeviceWidget.h"
#include <QWidget>
#include <QTimer>
#include <QSerialPort>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>


namespace Ui {
class DeviceWidget;
}

class DeviceWidget : public QWidget
{
    Q_OBJECT

public:
    
    explicit DeviceWidget(QWidget *parent = nullptr);
    ~DeviceWidget();
    void closeEvent(QCloseEvent*);
    void keyPressEvent(QKeyEvent* event) override; 
    void keyReleaseEvent(QKeyEvent* event) override; 
    void openSerialPort();
    void setOriginDevice();


public slots:
    void showData();
    void on_clicked_force_button();
    void on_clicked_record_button();

    void onReadyRead();              
    void sendCommand();              
signals:
    void ExitWin();

private:

    Ui::DeviceWidget ui;
    QTimer* timer = nullptr;

    HDErrorInfo error;
    HHD hHD;
    HDSchedulerHandle hForceCallback;
    DeviceInfo DeviceWidgetInfo;

    bool forceButton = false;
    bool recordButton = false;
    
    hduVector3Dd originDevicePos;
    hduVector3Dd originDeviceAngles;

    std::shared_ptr<spdlog::logger> deviceLog;
    int needleDepth = 0;
    QSerialPort* serial;
    QTimer* serialTimer = nullptr;
    QByteArray serialBuffer;             // 数据缓冲区
    void processLine(const QByteArray& line); // 解析单行数据
    bool footswitch;
};

#endif // DEVICEWIDGET_H
