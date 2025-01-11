#ifndef DEVICEWIDGET_H
#define DEVICEWIDGET_H

#include "UseDevice.h"

#include "ui_DeviceWidget.h"
#include <QWidget>
#include <QTimer>

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
    
   

public slots:
    void showData();
    void on_clicked_force_button();
    void on_clicked_record_button();

signals:
    void ExitWin();

private:
    Ui::DeviceWidget ui;
    QTimer* timer = nullptr;

    HDErrorInfo error;
    HHD hHD;
    HDSchedulerHandle hSphereCallback;
    DeviceInfo DeviceWidgetInfo;

    bool forceButton = false;
    bool recordButton = false;

    std::shared_ptr<spdlog::logger> deviceLog;

};

#endif // DEVICEWIDGET_H
